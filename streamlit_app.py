import streamlit as st
import pandas as pd
import io
import os
from datetime import datetime
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
    
    def get_workflow_status(row):
        app_statuses = [row[col] for col in app_columns]
        
        # If any application failed, the entire workflow failed
        if 'FAILURE' in app_statuses:
            return 'FAILED'
        
        # If any application is incomplete, the workflow is incomplete
        if 'INCOMPLETE' in app_statuses:
            return 'INCOMPLETE'
        
        # If all applications have no logs, workflow has no logs
        if all(status == 'NO_LOGS' for status in app_statuses):
            return 'NO_LOGS'
        
        # If all non-NO_LOGS applications are successful, workflow is successful
        non_no_logs_statuses = [status for status in app_statuses if status != 'NO_LOGS']
        if non_no_logs_statuses and all(status == 'SUCCESS' for status in non_no_logs_statuses):
            return 'SUCCESS'
        
        # Default case (mixed statuses without failures)
        return 'INCOMPLETE'
    
    # Apply the function to each row
    df['workflow_status'] = df.apply(get_workflow_status, axis=1)
    return df

def get_correlation_metrics(df):
    """Calculate metrics based on correlation ID status"""
    if 'workflow_status' not in df.columns:
        df = calculate_correlation_status(df)
    
    status_counts = df['workflow_status'].value_counts()
    
    return {
        'total_workflows': len(df),
        'successful_workflows': status_counts.get('SUCCESS', 0),
        'failed_workflows': status_counts.get('FAILED', 0),
        'incomplete_workflows': status_counts.get('INCOMPLETE', 0),
        'no_logs_workflows': status_counts.get('NO_LOGS', 0)
    }

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
                
                if st.button("🚀 Analyze Process", type="primary"):
                    if process_name:
                        with st.spinner(f"Analyzing process: {process_name}..."):
                            try:
                                if analyzer:
                                    # Real analysis
                                    df = analyzer.generate_process_report_batch(
                                        process_name, batch_size, max_results
                                    )
                                else:
                                    # Demo mode - use sample data
                                    st.warning("Demo mode: Using sample data")
                                    df = create_sample_data()
                                
                                if not df.empty:
                                    st.session_state['analysis_results'] = df
                                    st.session_state['process_name'] = process_name
                                    st.success(f"✅ Analysis completed for '{process_name}'!")
                                else:
                                    st.error("❌ No data found for the specified process")
                                    
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
                
                query_max_results = st.number_input("Max Results for Query:", min_value=1, max_value=10000, value=100)
                
                if st.button("🔍 Execute Query", type="primary"):
                    if spl_query:
                        with st.spinner("Executing SPL query..."):
                            try:
                                if analyzer:
                                    results = analyzer.run_custom_query(spl_query, query_max_results)
                                    if results:
                                        df = pd.DataFrame(results)
                                        st.session_state['query_results'] = df
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
            
            st.subheader(f"📈 Analysis Results for '{process_name}'")
            
            # Calculate correlation-based metrics
            metrics = get_correlation_metrics(df)
            
            # Summary metrics based on correlation IDs
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Workflows", metrics['total_workflows'])
            
            with col2:
                success_rate = (metrics['successful_workflows'] / metrics['total_workflows'] * 100) if metrics['total_workflows'] > 0 else 0
                st.metric("Successful Workflows", metrics['successful_workflows'], 
                         delta=f"{success_rate:.1f}%")
            
            with col3:
                failure_rate = (metrics['failed_workflows'] / metrics['total_workflows'] * 100) if metrics['total_workflows'] > 0 else 0
                st.metric("Failed Workflows", metrics['failed_workflows'], 
                         delta=f"{failure_rate:.1f}%")
            
            with col4:
                incomplete_rate = (metrics['incomplete_workflows'] / metrics['total_workflows'] * 100) if metrics['total_workflows'] > 0 else 0
                st.metric("Incomplete Workflows", metrics['incomplete_workflows'], 
                         delta=f"{incomplete_rate:.1f}%")
            
            with col5:
                no_logs_rate = (metrics['no_logs_workflows'] / metrics['total_workflows'] * 100) if metrics['total_workflows'] > 0 else 0
                st.metric("No Logs Workflows", metrics['no_logs_workflows'], 
                         delta=f"{no_logs_rate:.1f}%")
            
            # Add workflow status to DataFrame for display and download
            df_with_status = calculate_correlation_status(df.copy())
            
            # Download button
            csv_buffer = io.StringIO()
            df_with_status.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{process_name}_analysis_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download Analysis Report (CSV)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary"
            )
            
            # Display the dataframe with workflow status
            st.subheader("📊 Detailed Results")
            st.dataframe(df_with_status, use_container_width=True)
            
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
                
                # Workflow status summary
                st.subheader("📋 Workflow Status Summary")
                workflow_summary = df_with_status['workflow_status'].value_counts()
                
                # Create a nice display for workflow status
                status_colors = {
                    'SUCCESS': '🟢',
                    'FAILED': '🔴', 
                    'INCOMPLETE': '🟡',
                    'NO_LOGS': '⚪'
                }
                
                for status, count in workflow_summary.items():
                    percentage = (count / len(df_with_status)) * 100
                    color = status_colors.get(status, '⚫')
                    st.write(f"{color} **{status}**: {count} workflows ({percentage:.1f}%)")
        
        # Display query results
        elif 'query_results' in st.session_state:
            df = st.session_state['query_results']
            st.subheader("🔍 Query Results")
            
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
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Sample Workflows", sample_metrics['total_workflows'])
            with col2:
                st.metric("Sample Successful", sample_metrics['successful_workflows'])
            with col3:
                st.metric("Sample Failed", sample_metrics['failed_workflows'])
            with col4:
                st.metric("Sample Incomplete", sample_metrics['incomplete_workflows'])
            with col5:
                st.metric("Sample No Logs", sample_metrics['no_logs_workflows'])
            
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
