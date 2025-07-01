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
    page_title="Splunk Log Analysis System",
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
        st.markdown('<h1 class="main-header">📊 Splunk Log Analysis System</h1>', unsafe_allow_html=True)
        
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
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Workflows", len(df))
            
            if len(df.columns) > 1:  # Has application columns
                app_columns = [col for col in df.columns if col != 'correlation_id']
                
                with col2:
                    total_success = sum((df[col] == 'SUCCESS').sum() for col in app_columns)
                    st.metric("Total Successes", total_success)
                
                with col3:
                    total_failures = sum((df[col] == 'FAILURE').sum() for col in app_columns)
                    st.metric("Total Failures", total_failures)
                
                with col4:
                    total_incomplete = sum((df[col] == 'INCOMPLETE').sum() for col in app_columns)
                    st.metric("Total Incomplete", total_incomplete)
            
            # Download button
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
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
            
            # Display the dataframe
            st.subheader("📊 Detailed Results")
            st.dataframe(df, use_container_width=True)
            
            # Application-wise summary
            if len(df.columns) > 1:
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
            st.dataframe(sample_df, use_container_width=True)
            
            # Sample download button
            csv_buffer = io.StringIO()
            sample_df.to_csv(csv_buffer, index=False)
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
            "Splunk Log Analysis System"
            "</div>", 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()