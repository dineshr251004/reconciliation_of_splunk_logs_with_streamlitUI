import streamlit as st
import pandas as pd
import io
import os
from datetime import datetime, timedelta
import traceback

try:
    from main import SplunkLogAnalysis, SplunkConfig
    SPLUNK_AVAILABLE = True
except ImportError as e:
    st.error(f"Warning: Could not import SplunkLogAnalysis: {str(e)}")
    SPLUNK_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Process Reconciliation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #1f77b4;
    }
    .option-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .date-selector {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_analyzer():
    """Initialize the Splunk analyzer with caching"""
    if not SPLUNK_AVAILABLE:
        return None, "SplunkLogAnalysis module not available"
    
    try:
        splunk_config = SplunkConfig(
            host=os.getenv("SPLUNK_HOST", "localhost"),
            port=int(os.getenv("SPLUNK_PORT", 8089)),
            username=os.getenv("SPLUNK_USERNAME", "admin"),
            password=os.getenv("SPLUNK_PASSWORD", "changeme"),
            scheme=os.getenv("SPLUNK_SCHEME", "https"),
            index=os.getenv("SPLUNK_INDEX", "main"),
            verify=os.getenv("SPLUNK_VERIFY", "false").lower() == "true"
        )
        
        analyzer = SplunkLogAnalysis(splunk_config)
        return analyzer, None
    except Exception as e:
        return None, str(e)

def create_sample_data():
    """Create sample data for demonstration purposes"""
    sample_data = {
        'correlation_id': ['CORR_001', 'CORR_002', 'CORR_003', 'CORR_004', 'CORR_005'],
        'AuthService': ['SUCCESS', 'SUCCESS', 'FAILURE', 'SUCCESS', 'SUCCESS'],
        'PaymentGateway': ['SUCCESS', 'FAILURE', 'NO_LOGS', 'SUCCESS', 'INCOMPLETE'],
        'NotificationService': ['SUCCESS', 'SUCCESS', 'INCOMPLETE', 'SUCCESS', 'SUCCESS'],
        'OrderService': ['SUCCESS', 'SUCCESS', 'SUCCESS', 'FAILURE', 'SUCCESS']
    }
    return pd.DataFrame(sample_data)

def calculate_correlation_status(df):
    """
    Calculate the overall status for each correlation ID based on application statuses.
    Priority: FAILURE > INCOMPLETE > NO_LOGS > SUCCESS
    """
    app_columns = [col for col in df.columns if col != 'correlation_id']
    
    def get_corr_status(row):
        app_statuses = [row[col] for col in app_columns]
        
        # If any application failed, the entire process failed
        if 'FAILURE' in app_statuses:
            return 'FAILED'
        
        # If any application is incomplete, the process is incomplete
        if 'INCOMPLETE' in app_statuses:
            return 'INCOMPLETE'
        
        # If all applications have no logs, process has no logs
        if all(status == 'NO_LOGS' for status in app_statuses):
            return 'NO_LOGS'
        
        # If all non-NO_LOGS applications are successful, process is successful
        non_no_logs_statuses = [status for status in app_statuses if status != 'NO_LOGS']
        if non_no_logs_statuses and all(status == 'SUCCESS' for status in non_no_logs_statuses):
            return 'SUCCESS'
        
        # Default case (mixed statuses without failures)
        return 'INCOMPLETE'
    
    # Apply the function to each row
    df['overall_status'] = df.apply(get_corr_status, axis=1)
    return df

def get_correlation_metrics(df):
    """Calculate metrics based on correlation ID status"""
    if 'overall_status' not in df.columns:
        df = calculate_correlation_status(df)
    
    status_counts = df['overall_status'].value_counts()
    
    return {
        'total': len(df),
        'successful': status_counts.get('SUCCESS', 0),
        'failed': status_counts.get('FAILED', 0),
        'incomplete': status_counts.get('INCOMPLETE', 0) + status_counts.get('NO_LOGS', 0),
    }

def format_date_for_splunk(date_obj):
    """Format date for Splunk query - MM/DD/YYYY:HH:MM:SS format"""
    if date_obj:
        # Convert to MM/DD/YYYY:HH:MM:SS format
        return date_obj.strftime("%m/%d/%Y:%H:%M:%S")
    return None

def display_sample_logs():
    """Display sample log records"""
    sample_logs = [
        {
            '_time': '2024-01-15 10:30:25',
            'correlation_id': 'CORR_001',
            'application': 'AuthService',
            'process_name': 'user_authentication',
            'log_level': 'INFO',
            'message': 'User authentication successful',
            'user_id': 'user123'
        },
        {
            '_time': '2024-01-15 10:30:26',
            'correlation_id': 'CORR_001',
            'application': 'PaymentGateway',
            'process_name': 'payment_processing',
            'log_level': 'INFO',
            'message': 'Payment processed successfully',
            'details': 'Amount: $99.99, Card: ***1234'
        },
        {
            '_time': '2024-01-15 10:30:27',
            'correlation_id': 'CORR_002',
            'application': 'PaymentGateway',
            'process_name': 'payment_processing',
            'log_level': 'ERROR',
            'message': 'Payment processing failed',
            'details': 'Insufficient funds'
        }
    ]
    
    st.subheader("📋 Sample Log Records")
    st.dataframe(pd.DataFrame(sample_logs), use_container_width=True)

def highlight_overall_status_column(col):
    return [
        'color: red; font-weight: bold' if val == 'FAILED' else
        'color: orange; font-weight: bold' if val == 'INCOMPLETE' else
        'color: green; font-weight: bold' if val == 'SUCCESS' else
        'color: gray; font-weight: bold' if val == 'NO_LOGS' else
        ''
        for val in col
    ]

def main():
    try:
        # Header
        st.markdown('<h1 class="main-header">📊 End-to-End Process Reconciliation System</h1>', unsafe_allow_html=True)
        
        # Initialize analyzer
        with st.spinner("Initializing Splunk connection..."):
            analyzer, error = initialize_analyzer()
        
        if error:
            st.markdown(f'<div class="error-box">❌ <strong>Connection Error:</strong> {error}</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">💡 <strong>Demo Mode:</strong> Using sample data for demonstration</div>', unsafe_allow_html=True)
            analyzer = None
        else:
            st.markdown('<div class="success-box">✅ <strong>Connected to Splunk successfully!</strong></div>', unsafe_allow_html=True)
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            if analyzer:
                st.success("🟢 Splunk Connected")
            else:
                st.error("🔴 Using Demo Mode")
            
            st.subheader("Analysis Settings")
            batch_size = st.slider("Batch Size", min_value=1, max_value=20, value=5, 
                                  help="Number of correlation IDs to process in each batch")
            max_results = st.slider("Max Results", min_value=100, max_value=50000, value=10000, step=100,
                                   help="Maximum number of results to retrieve from Splunk")
            
            st.subheader("Splunk Configuration")
            st.info(f"""
            **Host:** {os.getenv('SPLUNK_HOST', 'localhost')}
            **Port:** {os.getenv('SPLUNK_PORT', '8089')}
            **Index:** {os.getenv('SPLUNK_INDEX', 'main')}
            """)
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            st.subheader("🔍 Choose Analysis Option")
            
            option = st.radio(
                "Select operation:",
                [
                    "1. Search for specific process",
                    "2. Run custom SPL query", 
                    "3. View system information"
                ],
                help="Choose the type of analysis you want to perform"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            st.subheader("📝 Input Parameters")
            
            if "1. Search for specific process" in option:
                process_name = st.text_input(
                    "Process Name:", 
                    placeholder="e.g., payment_processing, user_authentication",
                    help="Enter the name of the process you want to analyze"
                )
                
                # Date selection section
                st.subheader("📅 Date Range Selection")
                
                # Date range options
                date_option = st.selectbox(
                    "Select date range:",
                    ["All Time", "Today", "Yesterday", "Last 7 Days", "Last 30 Days", "Custom Range"],
                    help="Choose the time period for log analysis"
                )
                
                start_date = None
                end_date = None
                
                if date_option == "Today":
                    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
                elif date_option == "Yesterday":
                    yesterday = datetime.now() - timedelta(days=1)
                    start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
                elif date_option == "Last 7 Days":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                elif date_option == "Last 30 Days":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                elif date_option == "Custom Range":
                    col_start, col_end = st.columns(2)
                    with col_start:
                        start_date = st.date_input(
                            "Start Date:",
                            value=datetime.now().date() - timedelta(days=7),
                            help="Select the start date for analysis"
                        )
                        if start_date:
                            start_date = datetime.combine(start_date, datetime.min.time())
                    
                    with col_end:
                        end_date = st.date_input(
                            "End Date:",
                            value=datetime.now().date(),
                            help="Select the end date for analysis"
                        )
                        if end_date:
                            end_date = datetime.combine(end_date, datetime.max.time())
                
                # Display selected date range
                if start_date and end_date and date_option != "All Time":
                    st.info(f"📅 Selected Range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
                elif date_option == "All Time":
                    st.info("📅 Selected Range: All available data")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("🚀 Analyze Process", type="primary"):
                    if process_name:
                        # Format dates for Splunk - improved formatting
                        splunk_start_date = None
                        splunk_end_date = None
                        
                        if start_date and date_option != "All Time":
                            splunk_start_date = format_date_for_splunk(start_date)
                        if end_date and date_option != "All Time":
                            splunk_end_date = format_date_for_splunk(end_date)
                        
                        with st.spinner(f"Analyzing process: {process_name}..."):
                            try:
                                if analyzer:
                                    # Real analysis with date range
                                    df = analyzer.generate_process_report_batch(
                                        process_name, 
                                        batch_size, 
                                        max_results,
                                        start_date=splunk_start_date, 
                                        end_date=splunk_end_date
                                    )
                                else:
                                    # Demo mode - use sample data
                                    st.warning("Demo mode: Using sample data")
                                    df = create_sample_data()
                                
                                if not df.empty:
                                    st.session_state['analysis_results'] = df
                                    st.session_state['process_name'] = process_name
                                    st.session_state['date_range'] = date_option
                                    st.session_state['start_date'] = start_date
                                    st.session_state['end_date'] = end_date
                                    st.success(f"Analysis completed for '{process_name}'!")
                                else:
                                    st.error("❌ No data found for the specified process and date range")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during analysis: {str(e)}")
                                st.code(traceback.format_exc())
                    else:
                        st.warning("⚠️ Please enter a process name")
            
            elif "2. Run custom SPL query" in option:
                spl_query = st.text_area(
                    "SPL Query:",
                    placeholder="search index=main | stats count by process_name",
                    help="Enter your custom Splunk SPL query"
                )
                
                # Date selection for custom queries
                st.markdown('<div class="date-selector">', unsafe_allow_html=True)
                st.subheader("📅 Date Range for Query")
                
                query_date_option = st.selectbox(
                    "Select date range for query:",
                    ["All Time", "Today", "Yesterday", "Last 7 Days", "Last 30 Days", "Custom Range"],
                    help="Choose the time period for custom query"
                )
                
                query_start_date = None
                query_end_date = None
                
                if query_date_option == "Today":
                    query_start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    query_end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
                elif query_date_option == "Yesterday":
                    yesterday = datetime.now() - timedelta(days=1)
                    query_start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
                    query_end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
                elif query_date_option == "Last 7 Days":
                    query_end_date = datetime.now()
                    query_start_date = query_end_date - timedelta(days=7)
                elif query_date_option == "Last 30 Days":
                    query_end_date = datetime.now()
                    query_start_date = query_end_date - timedelta(days=30)
                elif query_date_option == "Custom Range":
                    col_start, col_end = st.columns(2)
                    with col_start:
                        query_start_date = st.date_input(
                            "Query Start Date:",
                            value=datetime.now().date() - timedelta(days=7),
                            key="query_start_date"
                        )
                        if query_start_date:
                            query_start_date = datetime.combine(query_start_date, datetime.min.time())
                    
                    with col_end:
                        query_end_date = st.date_input(
                            "Query End Date:",
                            value=datetime.now().date(),
                            key="query_end_date"
                        )
                        if query_end_date:
                            query_end_date = datetime.combine(query_end_date, datetime.max.time())
                
                # Display selected date range for query
                if query_start_date and query_end_date and query_date_option != "All Time":
                    st.info(f"📅 Query Range: {query_start_date.strftime('%Y-%m-%d %H:%M')} to {query_end_date.strftime('%Y-%m-%d %H:%M')}")
                elif query_date_option == "All Time":
                    st.info("📅 Query Range: All available data")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                query_max_results = st.number_input("Max Results for Query:", min_value=1, max_value=10000, value=100)
                
                if st.button("🔍 Execute Query", type="primary"):
                    if spl_query:
                        with st.spinner("Executing SPL query..."):
                            try:
                                if analyzer:
                                    # Add date range to query if specified
                                    modified_query = spl_query
                                    if query_start_date and query_end_date and query_date_option != "All Time":
                                        splunk_start = format_date_for_splunk(query_start_date)
                                        splunk_end = format_date_for_splunk(query_end_date)
                                        
                                        # Add date range to query if not already present
                                        if "earliest=" not in modified_query.lower() and "latest=" not in modified_query.lower():
                                            if modified_query.startswith("search "):
                                                modified_query = f'search earliest="{splunk_start}" latest="{splunk_end}" {modified_query[7:]}'
                                            else:
                                                modified_query = f'search earliest="{splunk_start}" latest="{splunk_end}" {modified_query}'
                                    
                                    results = analyzer.run_custom_query(modified_query, query_max_results)
                                    if results:
                                        df = pd.DataFrame(results)
                                        st.session_state['query_results'] = df
                                        st.session_state['query_date_range'] = query_date_option
                                        st.success(f"✅ Query executed successfully! Found {len(results)} results.")
                                    else:
                                        st.warning("⚠️ Query returned no results")
                                else:
                                    st.error("❌ Splunk connection not available in demo mode")
                            except Exception as e:
                                st.error(f"❌ Error executing query: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter an SPL query")
            
            else:  # Option 3
                st.info("📊 System Information")
                if analyzer:
                    try:
                        available_indexes = analyzer.splunk.get_available_indexes()
                        st.write("**Available Indexes:**")
                        st.write(available_indexes if available_indexes else "No indexes found")
                        
                        # Test connection
                        if st.button("🔧 Test Connection"):
                            with st.spinner("Testing connection..."):
                                result = analyzer.splunk.test_connection()
                                if result:
                                    st.success("✅ Connection test successful!")
                                else:
                                    st.error("❌ Connection test failed!")
                    except Exception as e:
                        st.error(f"❌ Error getting system info: {str(e)}")
                else:
                    st.warning("⚠️ No connection available - running in demo mode")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Results section
        st.markdown("---")
        
        # Display analysis results
        if 'analysis_results' in st.session_state:
            df = st.session_state['analysis_results']
            process_name = st.session_state.get('process_name', 'unknown')
            date_range = st.session_state.get('date_range', 'All Time')
            start_date = st.session_state.get('start_date')
            end_date = st.session_state.get('end_date')
            
            st.subheader(f"📈 Analysis Results for '{process_name}'")
            
            # Display date range info
            if date_range != "All Time" and start_date and end_date:
                st.info(f"📅 Date Range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.info("📅 Date Range: All available data")
            
            # Calculate correlation-based metrics
            metrics = get_correlation_metrics(df)
            
            # Summary metrics based on correlation IDs
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", metrics['total'])
            
            with col2:
                success_rate = (metrics['successful'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
                st.metric("Successful", metrics['successful'], 
                         delta=f"{success_rate:.1f}%")
            
            with col3:
                failure_rate = (metrics['failed'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
                st.metric("Failed", metrics['failed'], 
                         delta=f"-{failure_rate:.1f}%")
            
            with col4:
                incomplete_rate = (metrics['incomplete'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
                st.metric("Incomplete", metrics['incomplete'], 
                         delta=f"-{incomplete_rate:.1f}%")
            
            # Add process status to DataFrame for display and download
            df_with_status = calculate_correlation_status(df.copy())
            
            # Download button
            csv_buffer = io.StringIO()
            df_with_status.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{process_name}_analysis_{timestamp}.csv"
            
            st.download_button(
                label="Download Analysis Report (CSV)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary"
            )
            
            # Display the dataframe with process status
            st.subheader("📊 Detailed Results")
            styled_df = df_with_status.style.apply(
                            highlight_overall_status_column,
                            subset=['overall_status']
                        )

            st.dataframe(styled_df, use_container_width=True)
            
            # Application-wise summary
            if len(df.columns) > 1:
                app_columns = [col for col in df.columns if col != 'correlation_id']
                
                st.subheader("🔍 Application-wise Summary")
                for app in app_columns:
                    if app in df.columns:
                        app_summary = df[app].value_counts()
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.write(f"**{app}:**")
                        with col2:
                            st.write(app_summary.to_dict())
        
        # Display query results
        elif 'query_results' in st.session_state:
            df = st.session_state['query_results']
            query_date_range = st.session_state.get('query_date_range', 'All Time')
            
            st.subheader("🔍 Query Results")
            
            # Display date range info for query
            if query_date_range != "All Time":
                st.info(f"📅 Query Date Range: {query_date_range}")
            else:
                st.info("📅 Query Date Range: All available data")
            
            # Download button for query results
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"query_results_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download Query Results (CSV)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary"
            )
            
            st.dataframe(df, use_container_width=True)
        
        else:
            # Show sample data when no results are available
            st.subheader("📋 Sample Data Preview")
            st.info("💡 Run an analysis to see real results here. Below is sample data to show the expected format.")
            
            sample_df = create_sample_data()
            sample_df_with_status = calculate_correlation_status(sample_df)
            
            # Show sample metrics
            sample_metrics = get_correlation_metrics(sample_df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sample", sample_metrics['total'])
            with col2:
                st.metric("Sample Successful", sample_metrics['successful'])
            with col3:
                st.metric("Sample Failed", sample_metrics['failed'])
            with col4:
                st.metric("Sample Incomplete", sample_metrics['incomplete'])
            
            st.dataframe(sample_df_with_status, use_container_width=True)
            
            # Sample download button
            csv_buffer = io.StringIO()
            sample_df_with_status.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Sample Report (CSV)",
                data=csv_data,
                file_name="sample_analysis_report.csv",
                mime="text/csv"
            )
            
            display_sample_logs()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "End-to-End Process Reconciliation System"
            "</div>", 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
