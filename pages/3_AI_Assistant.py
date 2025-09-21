# pages/3_AI_Assistant.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from common import load_invoicing_report

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext, load_index_from_storage
import os

st.set_page_config(page_title="AI Data Assistant", layout="wide")
st.title("🤖 AI Data Assistant")

# Check if data is available
if "df" not in st.session_state or st.session_state.df is None:
    st.warning(
        "No data loaded. Please go to **Data Validation** and upload a CSV file first.")
    st.stop()

df = st.session_state.df

# Configure LlamaIndex


@st.cache_resource
def setup_llamaindex():
    """Setup LlamaIndex with OpenAI"""
    # You'll need to set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # Configure settings
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.node_parser = SimpleNodeParser.from_defaults(
        chunk_size=1000, chunk_overlap=200)

    return True


# Initialize LlamaIndex
setup_llamaindex()

# Create data analysis tools


def create_data_tools(df):
    """Create function tools for data analysis"""

    def get_data_overview():
        """Get comprehensive data overview"""
        overview = f"""
        Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
        Columns: {', '.join(df.columns.tolist())}
        Data Types: {df.dtypes.to_string()}
        Missing Values: {df.isnull().sum().to_string()}
        Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        Date Range: {df['Export Date'].min()} to {df['Export Date'].max()}
        Key Metrics:
        - Total GMV EUR: €{df['GMV EUR'].sum():,.2f}
        - Total TLG Fee: €{df['TLG Fee'].sum():,.2f}
        - Total COGS: €{df['COGS'].sum():,.2f}
        - Unique Orders: {df['Order Number'].nunique():,}
        - Unique Products: {df['Product ID'].nunique():,}
        """
        return overview

    def analyze_sales_trends():
        """Analyze sales trends and create visualizations"""
        try:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                daily_sales = df.groupby(df['Date'].dt.date)[
                    'GMV EUR'].sum().reset_index()
                daily_sales.columns = ['Date', 'GMV_EUR']

                # Create plot
                fig = px.line(daily_sales, x='Date', y='GMV_EUR',
                              title='Daily Sales Trend (GMV EUR)',
                              labels={'GMV_EUR': 'GMV EUR (€)', 'Date': 'Date'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                total_sales = df['GMV EUR'].sum()
                avg_daily = daily_sales['GMV_EUR'].mean()
                peak_day = daily_sales.loc[daily_sales['GMV_EUR'].idxmax()]

                return f"""
                Sales Trend Analysis:
                Total Sales: €{total_sales:,.2f}
                Average Daily Sales: €{avg_daily:,.2f}
                Peak Sales Day: {peak_day['Date']} (€{peak_day['GMV_EUR']:,.2f})
                """
        except Exception as e:
            return f"Error analyzing sales trends: {str(e)}"

    def get_top_products(n=5):
        """Get top N products by sales"""
        top_products = df.groupby('Product ID')['GMV EUR'].sum().nlargest(n)
        return f"Top {n} Products by Sales:\n{top_products.to_string()}"

    def get_top_countries(n=5):
        """Get top N countries by sales"""
        top_countries = df.groupby('Shipping Country')[
            'GMV EUR'].sum().nlargest(n)
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

        # Check for negative values
        if 'GMV EUR' in df.columns:
            negative_gmv = (df['GMV EUR'] < 0).sum()
            if negative_gmv > 0:
                validations.append(
                    f"⚠️ {negative_gmv} rows have negative GMV EUR")

        if 'TLG Fee' in df.columns:
            negative_tlg = (df['TLG Fee'] < 0).sum()
            if negative_tlg > 0:
                validations.append(
                    f"⚠️ {negative_tlg} rows have negative TLG Fee")

        # Check for missing values
        missing_critical = df[['Order Number',
                               'Product ID', 'GMV EUR']].isnull().sum()
        for col, count in missing_critical.items():
            if count > 0:
                validations.append(f"⚠️ {count} rows missing {col}")

        if validations:
            return "Data Validation Results:\n" + "\n".join(validations)
        else:
            return "✅ Data Validation Results: No issues found!"

    def filter_data(column, value, operation="equals"):
        """Filter data based on column and value"""
        try:
            if operation == "equals":
                filtered = df[df[column] == value]
            elif operation == "greater_than":
                filtered = df[df[column] > value]
            elif operation == "less_than":
                filtered = df[df[column] < value]
            elif operation == "contains":
                filtered = df[df[column].str.contains(str(value), na=False)]
            else:
                return f"Unknown operation: {operation}"

            return f"Filtered data ({len(filtered)} rows):\n{filtered.head(10).to_string()}"
        except Exception as e:
            return f"Error filtering data: {str(e)}"

    # Create function tools
    tools = [
        FunctionTool.from_defaults(
            fn=get_data_overview, name="get_data_overview"),
        FunctionTool.from_defaults(
            fn=analyze_sales_trends, name="analyze_sales_trends"),
        FunctionTool.from_defaults(
            fn=get_top_products, name="get_top_products"),
        FunctionTool.from_defaults(
            fn=get_top_countries, name="get_top_countries"),
        FunctionTool.from_defaults(fn=get_statistics, name="get_statistics"),
        FunctionTool.from_defaults(fn=validate_data, name="validate_data"),
        FunctionTool.from_defaults(fn=filter_data, name="filter_data"),
    ]

    return tools

# Create documents from dataframe


def create_data_documents(df):
    """Create documents from dataframe for vector store"""
    documents = []

    # Create overview document
    overview_text = f"""
    This is an invoicing dataset with {df.shape[0]} rows and {df.shape[1]} columns.
    Columns include: {', '.join(df.columns.tolist())}
    
    Key financial columns:
    - GMV EUR: Gross Merchandise Value in Euros
    - TLG Fee: The Luxury Group fee
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

    documents.append(Document(text=overview_text,
                     metadata={"type": "overview"}))

    # Create sample data document
    sample_text = f"""
    Sample data from the dataset:
    {df.head(5).to_string()}
    
    Data types:
    {df.dtypes.to_string()}
    """

    documents.append(Document(text=sample_text, metadata={"type": "sample"}))

    return documents

# Initialize or load index


@st.cache_resource
def get_or_create_index(df):
    """Get or create vector index for the dataframe"""
    try:
        # Try to load existing index
        storage_context = StorageContext.from_defaults(
            persist_dir="./data_index")
        index = load_index_from_storage(storage_context)
        return index
    except:
        # Create new index
        documents = create_data_documents(df)
        index = VectorStoreIndex.from_documents(documents)
        # Save index
        index.storage_context.persist(persist_dir="./data_index")
        return index


# Get tools and index
tools = create_data_tools(df)
index = get_or_create_index(df)

# Create query engine
query_engine = index.as_query_engine()

# Create agent
agent = ReActAgent.from_tools(tools, verbose=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the query with LlamaIndex agent
    with st.chat_message("assistant"):
        try:
            response = agent.chat(prompt)
            st.markdown(str(response))
            st.session_state.messages.append(
                {"role": "assistant", "content": str(response)})
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.markdown(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg})


# Add API key input
st.sidebar.header("🔑 OpenAI Configuration")
api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI features")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("✅ API key configured!")
else:
    st.warning(
        "⚠️ Please enter your OpenAI API key in the sidebar to use AI features.")
    st.stop()
