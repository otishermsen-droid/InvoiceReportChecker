# pages/4_Create_Reports.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from common import fetch_orders_returns

st.set_page_config(page_title="Create Reports", layout="wide")
st.title("📊 Create Reports from BigQuery")

# Check if data is available
if "df" not in st.session_state or st.session_state.df is None:
    st.warning(
        "No data loaded. Please go to **Data Validation** and upload a CSV file first.")
    st.stop()

# Sidebar for BigQuery configuration
st.sidebar.header("🔧 BigQuery Configuration")

# BigQuery settings
with st.sidebar.expander("⚙️ BigQuery Settings", expanded=False):
    bq_project = st.text_input(
        "Project ID",
        value="tlg-business-intelligence-prd",
        help="Your BigQuery project ID"
    )
    bq_dataset = st.text_input(
        "Dataset",
        value="bi",
        help="Your BigQuery dataset name"
    )
    bq_table = st.text_input(
        "Table",
        value="orders_returns_new",
        help="Your BigQuery table name"
    )

    # Update environment variables
    import os
    os.environ["BQ_PROJECT"] = bq_project
    os.environ["BQ_DATASET"] = bq_dataset
    os.environ["BQ_TABLE"] = bq_table

# Brand input
brand = st.sidebar.text_input(
    "Brand (2-letter acronym)",
    value="TL",
    help="Enter a 2-letter brand code (e.g., TL, LV, etc.)",
    max_chars=2
).upper()

# Date range inputs
st.sidebar.subheader("📅 Date Range")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=30),
        help="Select the start date for the query"
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        help="Select the end date for the query"
    )

# Validate date range
if start_date > end_date:
    st.sidebar.error("Start date must be before end date!")
    st.stop()

# Convert dates to string format DD-MM-YYYY
start_date_str = start_date.strftime("%d-%m-%Y")
end_date_str = end_date.strftime("%d-%m-%Y")

# Display selected parameters
st.sidebar.markdown("---")
st.sidebar.markdown("**Query Parameters:**")
st.sidebar.markdown(f"**Brand:** {brand}")
st.sidebar.markdown(f"**Start Date:** {start_date_str}")
st.sidebar.markdown(f"**End Date:** {end_date_str}")

# Main content area
st.markdown("## BigQuery Data Fetch")

# Query button
if st.button("🔍 Query BigQuery", type="primary", use_container_width=True):
    if not brand or len(brand) != 2:
        st.error("Please enter a valid 2-letter brand code!")
    else:
        with st.spinner("Fetching data from BigQuery..."):
            try:
                # Fetch data from BigQuery
                bq_df = fetch_orders_returns(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    brand=brand
                )

                if bq_df.empty:
                    st.warning("No data found for the specified criteria.")
                else:
                    # Store in session state
                    st.session_state.bq_df = bq_df
                    st.session_state.bq_params = {
                        "brand": brand,
                        "start_date": start_date_str,
                        "end_date": end_date_str
                    }

                    st.success(
                        f"✅ Successfully fetched {len(bq_df):,} rows from BigQuery!")

            except Exception as e:
                st.error(f"❌ Error fetching data from BigQuery: {str(e)}")
                st.info("Please check your BigQuery configuration and try again.")

# Display results if available
if "bq_df" in st.session_state and st.session_state.bq_df is not None:
    bq_df = st.session_state.bq_df
    params = st.session_state.bq_params

    st.markdown("---")
    st.markdown("## 📋 Query Results")

    # Display query info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(bq_df):,}")
    with col2:
        st.metric("Brand", params["brand"])
    with col3:
        st.metric("Date Range",
                  f"{params['start_date']} to {params['end_date']}")

    # Data preview
    st.markdown("### Data Preview")

    # Show basic info
    with st.expander("📊 Dataset Information", expanded=False):
        st.markdown(
            f"**Shape:** {bq_df.shape[0]:,} rows × {bq_df.shape[1]} columns")
        st.markdown(f"**Columns:** {', '.join(bq_df.columns.tolist())}")
        st.markdown(
            f"**Memory Usage:** {bq_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Data types
        st.markdown("**Data Types:**")
        st.dataframe(bq_df.dtypes.to_frame(
            'Data Type'), use_container_width=True)

    # Show the actual data
    st.markdown("### Raw Data")

    # Add filters
    col1, col2 = st.columns([3, 1])
    with col1:
        show_rows = st.slider("Number of rows to display",
                              10, min(100, len(bq_df)), 50)
    with col2:
        if st.button("🔄 Refresh Preview"):
            st.rerun()

    # Display the dataframe
    st.dataframe(
        bq_df.head(show_rows),
        use_container_width=True,
        height=400
    )

    # Download options
    st.markdown("### 💾 Download Options")
    col1, col2 = st.columns(2)

    with col1:
        csv_data = bq_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"bigquery_data_{params['brand']}_{params['start_date']}_to_{params['end_date']}.csv",
            mime="text/csv"
        )

    with col2:
        json_data = bq_df.to_json(
            orient='records', date_format='iso').encode('utf-8')
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"bigquery_data_{params['brand']}_{params['start_date']}_to_{params['end_date']}.json",
            mime="application/json"
        )

    # Data analysis
    st.markdown("### 📈 Quick Analysis")

    # Numeric columns analysis
    numeric_cols = bq_df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.markdown("**Numeric Columns Summary:**")
        st.dataframe(bq_df[numeric_cols].describe(), use_container_width=True)

    # Categorical columns analysis
    categorical_cols = bq_df.select_dtypes(
        include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.markdown("**Categorical Columns Summary:**")
        for col in categorical_cols[:5]:  # Show first 5 categorical columns
            if col in bq_df.columns:
                value_counts = bq_df[col].value_counts().head(10)
                st.markdown(f"**{col}:**")
                st.dataframe(value_counts.to_frame(
                    'Count'), use_container_width=True)

else:
    st.info("👆 Use the sidebar to configure your BigQuery query parameters and click 'Query BigQuery' to fetch data.")

    # Show example
    st.markdown("### 💡 Example Usage")
    st.markdown("""
    1. **Enter Brand Code**: Type a 2-letter brand code (e.g., "TL", "LV")
    2. **Select Date Range**: Choose start and end dates
    3. **Click Query BigQuery**: Fetch data from your BigQuery table
    4. **Preview Results**: View and analyze the returned data
    5. **Download Data**: Export as CSV or JSON if needed
    """)
