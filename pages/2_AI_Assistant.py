# pages/3_AI_Assistant.py
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# If you actually use this elsewhere, keep it; otherwise safe to remove
# from common import load_invoicing_report

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent

# ------------------------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="AI Data Assistant", layout="wide")
st.title("🤖 AI Data Assistant")

# ------------------------------------------------------------------------------------
# OpenAI API key (must be set BEFORE any LlamaIndex setup)
# ------------------------------------------------------------------------------------
st.sidebar.header("🔑 OpenAI Configuration")
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("❌ No OpenAI API key found in system environment variables. "
             "Please set OPENAI_API_KEY in your system properties.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
st.sidebar.success("✅ API key configured!")

# ------------------------------------------------------------------------------------
# Check if data is available
# ------------------------------------------------------------------------------------
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Please go to **Data Validation** and upload a CSV file first.")
    st.stop()

df = st.session_state.df

# ------------------------------------------------------------------------------------
# Configure LlamaIndex
# ------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def setup_llamaindex():
    """
    Configure global LlamaIndex settings.
    """
    # Model + embeddings
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    # Chunking
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    return True

setup_llamaindex()

# ------------------------------------------------------------------------------------
# Data analysis tools exposed to the agent
# ------------------------------------------------------------------------------------
def create_data_tools(df: pd.DataFrame):
    """Create function tools for data analysis over the in-memory DataFrame."""

    def get_data_overview():
        """Get comprehensive data overview"""
        # Defensive: stringify potential missing columns
        def col_safe(col):
            return df[col] if col in df.columns else pd.Series(dtype=object)

        lines = []
        lines.append(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        lines.append(f"Columns: {', '.join(df.columns.tolist())}")
        lines.append("Data Types:")
        lines.append(df.dtypes.to_string())
        lines.append("Missing Values:")
        lines.append(df.isnull().sum().to_string())
        lines.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        if "Export Date" in df.columns:
            try:
                dmin = pd.to_datetime(df["Export Date"]).min()
                dmax = pd.to_datetime(df["Export Date"]).max()
                lines.append(f"Date Range: {dmin} to {dmax}")
            except Exception:
                lines.append("Date Range: unavailable (could not parse 'Export Date')")
        else:
            lines.append("Date Range: N/A (no 'Export Date' column)")

        # Key metrics (guard if columns missing)
        total_gmv = df["GMV EUR"].sum() if "GMV EUR" in df.columns else np.nan
        total_tlg = df["TLG Fee"].sum() if "TLG Fee" in df.columns else np.nan
        total_cogs = df["COGS"].sum() if "COGS" in df.columns else np.nan
        uniq_orders = df["Order Number"].nunique() if "Order Number" in df.columns else 0
        uniq_products = df["Product ID"].nunique() if "Product ID" in df.columns else 0

        lines.append("Key Metrics:")
        lines.append(f"- Total GMV EUR: €{total_gmv:,.2f}" if pd.notna(total_gmv) else "- Total GMV EUR: N/A")
        lines.append(f"- Total TLG Fee: €{total_tlg:,.2f}" if pd.notna(total_tlg) else "- Total TLG Fee: N/A")
        lines.append(f"- Total COGS: €{total_cogs:,.2f}" if pd.notna(total_cogs) else "- Total COGS: N/A")
        lines.append(f"- Unique Orders: {uniq_orders:,}" if uniq_orders else "- Unique Orders: N/A")
        lines.append(f"- Unique Products: {uniq_products:,}" if uniq_products else "- Unique Products: N/A")

        return "\n".join(lines)

    def analyze_sales_trends():
        """Analyze sales trends and create visualizations"""
        try:
            if "Date" not in df.columns or "GMV EUR" not in df.columns:
                return "Cannot analyze sales trends: 'Date' and/or 'GMV EUR' column missing."

            # Work on a copy to avoid SettingWithCopy warnings
            tmp = df[["Date", "GMV EUR"]].copy()
            tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
            tmp = tmp.dropna(subset=["Date"])
            if tmp.empty:
                return "No valid dates found to compute sales trends."

            daily_sales = tmp.groupby(tmp["Date"].dt.date)["GMV EUR"].sum().reset_index()
            daily_sales.columns = ["Date", "GMV_EUR"]

            # Plot with Plotly
            fig = px.line(
                daily_sales,
                x="Date",
                y="GMV_EUR",
                title="Daily Sales Trend (GMV EUR)",
                labels={"GMV_EUR": "GMV EUR (€)", "Date": "Date"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

            total_sales = float(tmp["GMV EUR"].sum())
            avg_daily = float(daily_sales["GMV_EUR"].mean())
            peak_row = daily_sales.loc[daily_sales["GMV_EUR"].idxmax()]
            peak_day = peak_row["Date"]
            peak_val = float(peak_row["GMV_EUR"])

            return (
                "Sales Trend Analysis:\n"
                f"Total Sales: €{total_sales:,.2f}\n"
                f"Average Daily Sales: €{avg_daily:,.2f}\n"
                f"Peak Sales Day: {peak_day} (€{peak_val:,.2f})"
            )
        except Exception as e:
            return f"Error analyzing sales trends: {str(e)}"

    def get_top_products(n: int = 5):
        """Get top N products by sales"""
        if "Product ID" not in df.columns or "GMV EUR" not in df.columns:
            return "Missing required columns: 'Product ID' and/or 'GMV EUR'."
        top_products = df.groupby("Product ID")["GMV EUR"].sum().nlargest(n)
        return f"Top {n} Products by Sales:\n{top_products.to_string()}"

    def get_top_countries(n: int = 5):
        """Get top N countries by sales"""
        if "Shipping Country" not in df.columns or "GMV EUR" not in df.columns:
            return "Missing required columns: 'Shipping Country' and/or 'GMV EUR'."
        top_countries = df.groupby("Shipping Country")["GMV EUR"].sum().nlargest(n)
        return f"Top {n} Countries by Sales:\n{top_countries.to_string()}"

    def get_statistics():
        """Get statistical analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return "No numeric columns found for statistical analysis."
        stats = df[numeric_cols].describe()
        return f"Statistical Analysis:\n{stats.to_string()}"

    def validate_data():
        """Perform data validation checks"""
        validations = []

        # Negative values
        if "GMV EUR" in df.columns:
            negative_gmv = (df["GMV EUR"] < 0).sum()
            if negative_gmv > 0:
                validations.append(f"⚠️ {negative_gmv} rows have negative GMV EUR")

        if "TLG Fee" in df.columns:
            negative_tlg = (df["TLG Fee"] < 0).sum()
            if negative_tlg > 0:
                validations.append(f"⚠️ {negative_tlg} rows have negative TLG Fee")

        # Missing critical fields
        critical = [c for c in ["Order Number", "Product ID", "GMV EUR"] if c in df.columns]
        if critical:
            missing_critical = df[critical].isnull().sum()
            for col, count in missing_critical.items():
                if count > 0:
                    validations.append(f"⚠️ {count} rows missing {col}")

        if validations:
            return "Data Validation Results:\n" + "\n".join(validations)
        else:
            return "✅ Data Validation Results: No issues found!"

    def filter_data(column, value, operation: str = "equals"):
        """Filter data based on column and value"""
        if column not in df.columns:
            return f"Unknown column: {column}"

        try:
            if operation == "equals":
                filtered = df[df[column] == value]
            elif operation == "greater_than":
                filtered = df[df[column] > value]
            elif operation == "less_than":
                filtered = df[df[column] < value]
            elif operation == "contains":
                filtered = df[df[column].astype(str).str.contains(str(value), na=False)]
            else:
                return f"Unknown operation: {operation}"

            head = filtered.head(10)
            return f"Filtered data ({len(filtered)} rows):\n{head.to_string()}"
        except Exception as e:
            return f"Error filtering data: {str(e)}"

    tools = [
        FunctionTool.from_defaults(fn=get_data_overview, name="get_data_overview"),
        FunctionTool.from_defaults(fn=analyze_sales_trends, name="analyze_sales_trends"),
        FunctionTool.from_defaults(fn=get_top_products, name="get_top_products"),
        FunctionTool.from_defaults(fn=get_top_countries, name="get_top_countries"),
        FunctionTool.from_defaults(fn=get_statistics, name="get_statistics"),
        FunctionTool.from_defaults(fn=validate_data, name="validate_data"),
        FunctionTool.from_defaults(fn=filter_data, name="filter_data"),
    ]
    return tools

tools = create_data_tools(df)

# ------------------------------------------------------------------------------------
# Build / load a tiny vector index (for background context to the agent)
# ------------------------------------------------------------------------------------
def create_data_documents(df: pd.DataFrame):
    """Create lightweight documents derived from the DataFrame for retrieval."""
    docs = []

    overview_text = f"""
    This is an invoicing dataset with {df.shape[0]} rows and {df.shape[1]} columns.
    Columns include: {', '.join(df.columns.tolist())}

    Key financial columns:
    - GMV EUR: Gross Merchandise Value in Euros
    - TLG Fee: The Level Group fee
    - COGS: Cost of Goods Sold
    - GMV Net VAT: GMV after VAT

    Date columns:
    - Export Date: When the data was exported
    - Date: Transaction date

    Other important columns:
    - Order Number: Unique order identifier
    - Product ID: Product identifier
    - Shipping Country: Destination country
    - Store ID: Store identifier
    """
    docs.append(Document(text=overview_text, metadata={"type": "overview"}))

    sample_text = f"""
    Sample data from the dataset:
    {df.head(5).to_string()}

    Data types:
    {df.dtypes.to_string()}
    """
    docs.append(Document(text=sample_text, metadata={"type": "sample"}))

    return docs

@st.cache_resource(show_spinner=False)
def get_or_create_index(df: pd.DataFrame):
    """Get or create vector index for the dataframe (persisted locally)."""
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./data_index")
        index = load_index_from_storage(storage_context)
        return index
    except Exception:
        documents = create_data_documents(df)
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="./data_index")
        return index

index = get_or_create_index(df)

# (Optional) a query engine if you want to retrieve context separately
query_engine = index.as_query_engine()

# ------------------------------------------------------------------------------------
# Create the agent (OpenAI-backed), wiring in the tools
# ------------------------------------------------------------------------------------
agent = OpenAIAgent.from_tools(tools, llm=Settings.llm, verbose=True)

# ------------------------------------------------------------------------------------
# Simple chat UI with history
# ------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with the agent
    with st.chat_message("assistant"):
        try:
            response = agent.chat(prompt)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
