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

# Sidebar for BigQuery configuration (header only, settings removed)
st.sidebar.header("🔧 BigQuery Configuration")

# Brand input
brand = st.sidebar.text_input(
    "Brand (2-letter acronym)",
    value="PJ",
    help="Enter a 2-letter brand code (e.g. DG, SC)",
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
if st.button("🔍 Query BigQuery", type="primary", width='stretch'):
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

    # Show the actual data
    st.markdown("#### Preview Data")

    # Display the dataframe
    st.dataframe(
        bq_df.head(50),
        width='stretch',
        height=400
    )

    csv_data = bq_df.to_csv(index=False).encode('utf-8-sig')

    # Data analysis
    st.markdown("### 👀 BQ Cross Check")

    st.markdown("#### Units sold")

    st.markdown("#### Orders")

    st.markdown("#### Returns")

    st.markdown("### 🌍 DDP Check")


else:
    st.info("👆 Use the sidebar to configure your BigQuery query parameters and click 'Query BigQuery' to fetch data.")