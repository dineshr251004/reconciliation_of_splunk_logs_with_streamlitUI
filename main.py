import re
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
from langchain_ollama import OllamaLLM
from dataclasses import dataclass
import time
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import jellyfish 
from dotenv import load_dotenv
import os
import csv
import io

import splunklib.client as client

load_dotenv()

@dataclass
class SplunkConfig:
    """Configuration for Splunk connection"""
    host: str 
    port: int
    username: str 
    password: str 
    scheme: str 
    index: str 
    verify: bool 

class ProcessMatcher:
    """
    Enhanced process matching system that combines multiple similarity metrics
    for better matching accuracy without hardcoding process names.
    """
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = self.preprocess_text(text).split()
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords
    
    def calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """Calculate lexical similarity using multiple string matching techniques"""
        text1_clean = self.preprocess_text(text1)
        text2_clean = self.preprocess_text(text2)
        
        # Sequence Matcher
        seq_sim = SequenceMatcher(None, text1_clean, text2_clean).ratio()
        
        # Jaro-Winkler similarity
        jaro_sim = jellyfish.jaro_winkler_similarity(text1_clean, text2_clean)
        
        # Levenshtein distance
        max_len = max(len(text1_clean), len(text2_clean))
        if max_len == 0:
            leven_sim = 1.0
        else:
            leven_sim = 1 - (jellyfish.levenshtein_distance(text1_clean, text2_clean) / max_len)
        
        # Keyword overlap
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 and not keywords2:
            keyword_sim = 1.0
        elif not keywords1 or not keywords2:
            keyword_sim = 0.0
        else:
            intersection = len(keywords1.intersection(keywords2))
            union = len(keywords1.union(keywords2))
            keyword_sim = intersection / union if union > 0 else 0.0
        
        combined_score = (
            seq_sim * 0.3 +
            jaro_sim * 0.3 +
            leven_sim * 0.2 +
            keyword_sim * 0.2
        )
        
        return combined_score
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            embedding1 = self.embedding_model.encode([self.preprocess_text(text1)])
            embedding2 = self.embedding_model.encode([self.preprocess_text(text2)])
            return cosine_similarity(embedding1, embedding2)[0][0]
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def find_best_matches(self, user_input: str, available_processes: List[str]) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Find best matching processes using combined similarity metrics
        
        Returns:
            List of tuples (process_name, combined_score, individual_scores)
        """
        results = []
        
        for process in available_processes:
            lexical_sim = self.calculate_lexical_similarity(user_input, process)
            semantic_sim = self.calculate_semantic_similarity(user_input, process)
            
            combined_score = (
                lexical_sim * 0.6 +
                semantic_sim * 0.4 
            )
            
            individual_scores = {
                'lexical': lexical_sim,
                'semantic': semantic_sim,
                'combined': combined_score
            }
            
            results.append((process, combined_score, individual_scores))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:1]


class SPLQueryGenerator:
    """Generate SPL queries based on user input using LLM"""
    
    def __init__(self, llm_model: str = "mistral:7b"):
        self.llm = OllamaLLM(model=llm_model)
    
    @staticmethod
    def validate_and_fix_spl_query(query: str) -> str:
        """Validate and fix common SPL query syntax issues"""
        query = query.strip()
        
        # If query doesn't start with 'search' or '|', add 'search' prefix
        if not query.startswith(('search ', '|')):
            if 'index=' in query:
                query = f'search {query}'
            else:
                query = f'| {query}'
        
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def generate_process_query(self, process_name: str, time_range: str = "0", 
                        index: str = "main", max_results: int = 10000, 
                        start_date: str = None, end_date: str = None) -> str:
        """Generate SPL query to find logs for a specific process with optional date range"""
        
        # Handle date range - improved logic
        time_clause = ""
        if start_date and end_date:
            time_clause = f'earliest="{start_date}" latest="{end_date}"'
        elif start_date:
            time_clause = f'earliest="{start_date}"'
        elif end_date:
            time_clause = f'latest="{end_date}"'
        elif time_range != "0":
            time_clause = f"earliest={time_range}"
        else:
            time_clause = "earliest=0"
        
        # Build the query with proper time clause integration
        spl_query = f'''search index={index} {time_clause} 
                        (process_name="*{process_name}*" OR message="*{process_name}*" OR _raw="*{process_name}*") 
                        | dedup correlation_id 
                        | map maxsearches={max_results} search="search index={index} earliest=0 correlation_id=\\"$correlation_id$\\" | table _time, correlation_id, application, process_name, log_level, message, details, user_id, session_id, request_id, host"
                        | sort correlation_id, _time
                    '''
        
        return spl_query


class SplunkSDKConnector:
    """Connector to interact with Splunk using the official Splunk SDK"""
    
    def __init__(self, config: SplunkConfig):
        self.config = config
        self.service = None
        self.connected = False
    
    def connect(self) -> bool:
        """Establish connection to Splunk using SDK"""
        try:
            print(f"Connecting to Splunk at {self.config.host}:{self.config.port}")
            
            # Create connection using Splunk SDK
            self.service = client.connect(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                scheme=self.config.scheme
            )
            
            # Test connection
            info = self.service.info
            print(f"Successfully connected to Splunk server version: {info.get('version', 'Unknown')}")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Failed to connect to Splunk: {e}")
            self.connected = False
            return False
    
    def execute_search(self, spl_query: str, max_results: int = 10000, 
                      timeout: int = 300) -> List[Dict]:
        """Execute SPL query using Splunk SDK and return results"""
        
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            # Validate and fix the query
            spl_query = SPLQueryGenerator.validate_and_fix_spl_query(spl_query)
            
            # Create search job using SDK
            search_kwargs = {
                'search_mode': 'normal',
                'output_mode': 'json',
                'count': 0  
            }
            
            job = self.service.jobs.create(spl_query, **search_kwargs)
            print(f"Search job created with SID: {job.sid}")
            
            # Wait for job completion with timeout and proper status checking
            start_time = time.time()
            while True:
                job.refresh()  
                
                # Check if job is done
                if job.is_done():
                    print("Search completed successfully")
                    break
                
                # Check for job failure
                if job.state in ['FAILED', 'FINALIZING']:
                    print(f"Search job failed with state: {job.state}")
                    try:
                        job.refresh()
                        messages = job.content.get('messages', [])
                        if messages:
                            print("Error messages:")
                            for msg in messages:
                                print(f"  {msg}")
                    except:
                        pass
                    
                    try:
                        job.cancel()
                    except:
                        pass
                    return []
                
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    print(f"Search timeout after {timeout} seconds")
                    try:
                        job.cancel()
                    except:
                        pass
                    return []
                
                try:
                    progress = float(job.content.get("doneProgress", 0)) * 100
                    print(f"Search progress: {progress:.1f}%")
                except:
                    print("Search in progress...")
                
                time.sleep(2)
            
            # Job statistics
            try:
                job.refresh()
                job_stats = {}
                job_stats['event_count'] = job.content.get("eventCount", "Unknown")
                job_stats['result_count'] = job.content.get("resultCount", "Unknown")
                job_stats['scan_count'] = job.content.get("scanCount", "Unknown")
                job_stats['run_duration'] = job.content.get("runDuration", "Unknown")
                print(f"Job statistics: {job_stats}")
                
            except Exception as stats_error:
                print(f"Could not get job statistics: {stats_error}")
            
            # Get results using JSON output mode
            results_list = []
            result_count = 0
            
            try:
                print("Retrieving search results...")
                
                results_args = {
                    'output_mode': 'json',
                    'count': 0 
                }
                
                # Get raw results
                raw_results = job.results(**results_args)
                
                # Parse JSON results directly
                results_data = raw_results.read()
                
                if isinstance(results_data, bytes):
                    results_data = results_data.decode('utf-8')
                
                # Handle different JSON formats that Splunk might return
                try:
                    # Parse as a single JSON object 
                    parsed_data = json.loads(results_data)
                    
                    if isinstance(parsed_data, dict):
                        # Handle different response formats
                        if 'results' in parsed_data:
                            if isinstance(parsed_data['results'], list):
                                results_list = parsed_data['results'][:max_results]
                            elif isinstance(parsed_data['results'], str):
                                results_str = parsed_data['results']
                                try:
                                    results_list = json.loads(results_str)
                                    if isinstance(results_list, list):
                                        results_list = results_list[:max_results]
                                    else:
                                        results_list = [results_list]
                                except:
                                    print("Could not parse results string")
                        elif 'preview' in parsed_data or '_time' in parsed_data:
                            results_list = [parsed_data]
                    elif isinstance(parsed_data, list):
                        results_list = parsed_data[:max_results]
                        
                except json.JSONDecodeError:
                    # Parse data from JSONL format
                    for line in results_data.strip().split('\n'):
                        if result_count >= max_results:
                            break
                        
                        try:
                            if line.strip():
                                result = json.loads(line)
                                
                                # Handle different result types
                                if isinstance(result, dict):
                                    clean_result = {}
                                    for key, value in result.items():
                                        # Handle multi-value fields
                                        if isinstance(value, list):
                                            clean_result[key] = ', '.join(str(v) for v in value)
                                        else:
                                            clean_result[key] = str(value) if value is not None else ''
                                    
                                    results_list.append(clean_result)
                                    result_count += 1
                        except json.JSONDecodeError:
                            continue
                        except Exception as parse_error:
                            print(f"Error parsing result line: {parse_error}")
                            continue
                
                print(f"Successfully retrieved {len(results_list)} results from Splunk")
                
            except Exception as results_error:
                print(f"Error retrieving results: {results_error}")
                print("Trying alternative result retrieval method...")
                
                try:
                    csv_results = job.results(output_mode='csv', count=0)
                    
                    # Read as text first
                    results_text = csv_results.read()
                    if isinstance(results_text, bytes):
                        results_text = results_text.decode('utf-8')
                   
                    csv_reader = csv.DictReader(io.StringIO(results_text))
                    results_list = []
                    for row in csv_reader:
                        if len(results_list) >= max_results:
                            break
                        results_list.append(dict(row))
                    
                    print(f"Fallback CSV method retrieved {len(results_list)} results")
                
                except Exception as fallback_error:
                    print(f"Fallback method also failed: {fallback_error}")
            
            # Clean up job
            try:
                job.cancel()
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up job: {cleanup_error}")
            
            return results_list
            
        except Exception as e:
            print(f"Error executing search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_connection(self) -> bool:
        """Test the Splunk connection with a very simple query"""
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # Simple test query
            test_query = "| makeresults | eval test=\"connection_test\""
            
            results = self.execute_search(test_query, max_results=1)
            
            if results and len(results) > 0:
                print(f"Connection test successful. Test query returned {len(results)} result(s).")
                print(f"Now checking for the existence of target index")

                # Check if the target index exists
                index_query = f"| rest /services/data/indexes | search title=\"{self.config.index}\" | table title"
                index_results = self.execute_search(index_query, max_results=1)
                
                if index_results:
                    print(f"Index '{self.config.index}' exists and is accessible")
                else:
                    print(f"Index '{self.config.index}' might not exist or is not accessible")
                    print("Available indexes:")
                    available_indexes = self.get_available_indexes()
                    print(f"  {available_indexes}")
                
                return True
            else:
                print("Connection test failed - no results returned")
                return False
                
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_available_indexes(self) -> List[str]:
        """Get list of available indexes"""
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            indexes = []
            for index in self.service.indexes:
                indexes.append(index.name)
            return sorted(indexes)
        except Exception as e:
            print(f"Error getting indexes: {e}")
            return []

class SplunkLogAnalysis:
    """Enhanced Splunk log analysis with LLM capabilities"""
    
    def __init__(self, splunk_config: SplunkConfig, llm_model: str = "mistral:7b", model_name="all-MiniLM-L6-v2"):
        self.splunk = SplunkSDKConnector(splunk_config)
        self.query_generator = SPLQueryGenerator(llm_model)
        self.llm = OllamaLLM(model=llm_model)
        self.applications = set()
        self.embedding_model = SentenceTransformer(model_name)
        self.process_matcher = ProcessMatcher(self.embedding_model)

        # Test connection on initialization
        print("Testing Splunk connection...")
        connection_result = self.splunk.test_connection()
        if connection_result:
            print("Splunk connection successful")
        else:
            print("Warning: Could not establish reliable connection to Splunk")
    
    def run_custom_query(self, spl_query: str, max_results: int = 1000) -> List[Dict]:
        """Run a custom SPL query and return raw results"""
        return self.splunk.execute_search(spl_query, max_results)
    
    def get_available_processes(self) -> List[str]:
        """Get list of available process names from Splunk"""
        query = f"search index={self.splunk.config.index} earliest=0 | stats count by process_name | sort -count | head 50"
        results = self.run_custom_query(query)
        
        process_names = []
        for result in results:
            if 'process_name' in result and result['process_name']:
                base_name = result['process_name'].split('_')[0]
                if base_name not in process_names:
                    process_names.append(base_name)
        return process_names
    
    def find_process_matches(self, user_input: str) -> List[str]:
        """Find matching processes in Splunk data"""
        available_processes = self.get_available_processes()
        match = self.process_matcher.find_best_matches(user_input,available_processes)
        return match[0]
    
    def retrieve_logs_by_process(self, process_name: str, max_results: int = 10000, 
                            start_date: str = None, end_date: str = None) -> Dict[str, List[Dict]]:
        """Retrieve logs for a specific process, grouped by correlation ID"""
        print(f"Retrieving logs for process: {process_name}")
        
        if start_date or end_date:
            print(f"Date range: {start_date} to {end_date}")
        
        # Query for all logs related to this process
        query = self.query_generator.generate_process_query(
            process_name, 
            index=self.splunk.config.index,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date
        )
        
        results = self.run_custom_query(query, max_results)
        
        # Group by correlation ID
        logs_by_correlation = defaultdict(list)
        for result in results:
            correlation_id = result.get('correlation_id', 'unknown')
            logs_by_correlation[correlation_id].append(result)
            
            # Track applications
            app = result.get('application', 'unknown')
            if app:
                self.applications.add(app)     
        
        # Sort logs within each correlation ID by timestamp
        for correlation_id in logs_by_correlation:
            logs_by_correlation[correlation_id].sort(
                key=lambda x: x.get('_time', '0')
            )
        
        print(f"Found logs for {len(logs_by_correlation)} correlation IDs")
        return dict(logs_by_correlation)
    
    def analyze_logs_batch_with_llm(self, batch_data: Dict[str, List[Dict]], 
                                   process_name: str, batch_size: int = 5) -> Dict[str, Dict[str, str]]:
        """Analyze multiple correlation IDs in a single LLM call for efficiency"""
        batch_results = {}
        correlation_ids = list(batch_data.keys())
        
        for i in range(0, len(correlation_ids), batch_size):
            batch_correlation_ids = correlation_ids[i:i + batch_size]
            batch_context = self._prepare_batch_context_for_llm(
                {cid: batch_data[cid] for cid in batch_correlation_ids}
            )
            
            print(f"Processing batch {i//batch_size + 1}: {len(batch_correlation_ids)} correlation IDs")
            
            # Generate batch prompt
            prompt = self._create_batch_prompt(batch_correlation_ids, batch_context, process_name)
            
            try:
                response = self.llm.invoke(prompt).strip()
                batch_parsed_results = self._parse_batch_response(response, batch_correlation_ids)
                batch_results.update(batch_parsed_results)
                
            except Exception as e:
                print(f"Error analyzing batch {i//batch_size + 1}: {e}")
                for cid in batch_correlation_ids:
                    batch_results[cid] = {app: "NO_LOGS" for app in self.applications}
        
        return batch_results
    
    def _prepare_batch_context_for_llm(self, batch_data: Dict[str, List[Dict]]) -> str:
        """Prepare context for multiple correlation IDs"""
        context_sections = []
        
        for correlation_id, logs in batch_data.items():
            if not logs:
                context_sections.append(f"\n--- CORRELATION_ID: {correlation_id} ---")
                context_sections.append("NO LOGS FOUND")
                continue
            
            context_sections.append(f"\n--- CORRELATION_ID: {correlation_id} ---")
            
            for log_data in logs:
                details = log_data.get('details', '')
                if len(details) > 100:
                    details = details[:100] + "..."
                
                signals = []
                message_lower = log_data.get('message', '').lower()
                details_lower = details.lower() if details else ""
                
                combined_text = f"{message_lower} {details_lower}"
                
                success_signals = ["accepted", "completed", "success", "authorized", "initiated"]
                failure_signals = ["failed", "rejected", "declined", "not authorized", "timeout", "exception", "error"]
                
                for word in success_signals:
                    if word in combined_text:
                        signals.append(f"success {word}")
                
                for word in failure_signals:
                    if word in combined_text:
                        signals.append(f"failure/warn/incomplete {word}")
                
                # Create log line with signals
                timestamp = log_data.get('_time', log_data.get('timestamp', ''))
                application = log_data.get('application', 'unknown')
                log_level = log_data.get('log_level', 'INFO')
                message = log_data.get('message', '')
                
                log_line = f"[{application}] [{timestamp}] {log_level}: {message}"
                if details:
                    log_line += f" | {details}"
                
                if signals:
                    log_line += f" | SIGNALS: {'; '.join(signals)}"
                
                context_sections.append(log_line)
        
        return "\n".join(context_sections)
    
    def _create_batch_prompt(self, correlation_ids: List[str], context: str, process_name: str) -> str:
        """Create prompt for batch analysis"""
        correlation_list = ", ".join(correlation_ids)
        
        prompt = f"""
            You are a highly accurate system log analysis engine.

            Your task is to analyze the following end-to-end logs for the process: **{process_name}**.

            You are analyzing multiple correlation IDs: {correlation_list}

            ---
            Each workflow has logs from multiple applications (like AuthService, PaymentGateway, etc).

            For each correlation ID, examine the logs and assign a **status** to each application involved in that workflow.

            Use these **ONLY**: `SUCCESS`, `FAILURE`, `INCOMPLETE`, `NO_LOGS`

            **SUCCESS**: If the logs indicate the application completed its step or processed without errors.
            **FAILURE**: If there are errors, exceptions, or failed operations logged.
            **INCOMPLETE**: If logs are present but the execution does not seem to finish properly.
            **NO_LOGS**: If logs from that application are completely missing.

            ---
            Your response format MUST strictly be:

            CORRELATION_ID_1:
            AuthService: SUCCESS
            PaymentGateway: FAILURE
            ...

            CORRELATION_ID_2:
            NotificationService: INCOMPLETE
            ...

            If no logs are present for a correlation ID, output:

            CORRELATION_ID_3:
            NO_LOGS

            ---
            LOGS:
            {context}
            """
        return prompt
    
    def _parse_batch_response(self, response: str, correlation_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """Parse LLM response for batch analysis"""
        results = {}
        
        # Initialize all correlation IDs with NO_LOGS for all apps
        for cid in correlation_ids:
            results[cid] = {app: "NO_LOGS" for app in self.applications}
        
        lines = response.split('\n')
        current_correlation_id = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a correlation ID header
            found_correlation = None
            for cid in correlation_ids:
                if cid in line and ':' in line and line.endswith(':'):
                    found_correlation = cid
                    break
            
            if found_correlation:
                current_correlation_id = found_correlation
                continue
            
            # Parse application status if we have a current correlation ID
            if current_correlation_id and ':' in line:
                for app in self.applications:
                    if app.lower() in line.lower():
                        status_match = re.search(rf"{re.escape(app)}[:\- ]+\s*(SUCCESS|FAILURE|INCOMPLETE|NO_LOGS)", 
                                               line, re.IGNORECASE)
                        if status_match:
                            status = status_match.group(1).upper()
                            results[current_correlation_id][app] = status
                            break
        
        return results
    
    def analyze_process_by_name_batch(self, user_input: str, batch_size: int = 5, max_results: int = 10000, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Analyze process using batch processing with Splunk data"""
        print(f"\n")
        print(f"BATCH ANALYZING PROCESS: {user_input.upper()} (Batch Size: {batch_size})")
        
        # Find matching process names
        print(f"Finding matching process names")
        matching_processes = self.find_process_matches(user_input)
        
        if not matching_processes:
            print(f"No processes found matching: {user_input}")
            return pd.DataFrame()
        
        # Best match for analysis
        best_match = matching_processes[0]
        print(f"\nBest match: {best_match}")
        # Retrieve logs for the best matching process
        all_correlation_logs = self.retrieve_logs_by_process(best_match, max_results,start_date=start_date,end_date=end_date)
        
        if not all_correlation_logs:
            print("No logs found for matching process")
            return pd.DataFrame()
        
        # Determine involved applications
        involved_apps = list(self.applications)
        print(f"Involved applications: {involved_apps}")
        print(f"Processing {len(all_correlation_logs)} correlation IDs in batches of {batch_size}...")
        
        # Analysis using LLM
        batch_results = self.analyze_logs_batch_with_llm(all_correlation_logs, best_match, batch_size)
        
        # Convert results to DataFrame format
        all_results = []
        for correlation_id in all_correlation_logs.keys():
            row = {"correlation_id": correlation_id}
            
            correlation_results = batch_results.get(correlation_id, {})
            
            for app_name in involved_apps:
                row[app_name] = correlation_results.get(app_name, "NO_LOGS")
            
            all_results.append(row)
        
        df = pd.DataFrame(all_results)
        df.attrs["involved_apps"] = involved_apps  # Store for later use in summary and CSV
        return df
    
    def generate_process_report_batch(self, user_input: str, batch_size: int = 5, 
                                    max_results: int = 10000, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Generate batch analysis report for a process"""
        df = self.analyze_process_by_name_batch(user_input, batch_size, max_results, start_date= start_date, end_date=end_date)
        
        if df.empty:
            return df
        
        involved_apps = getattr(df, 'attrs', {}).get('involved_apps', [])
        
        if involved_apps:
            columns_to_save = ['correlation_id'] + involved_apps
            df_filtered = df[columns_to_save].copy()
        else:
            df_filtered = df.copy()
        
        return df_filtered
      
