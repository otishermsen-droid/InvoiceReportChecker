# pages/2_🤖_Assistant.py
import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent

st.set_page_config(page_title="Assistant", layout="wide")
st.title("Assistant (experimental)")

# Check inputs from previous page(s)
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("No CSV loaded yet. Go back to **Home** and upload a file.")
    st.stop()

if "cogs_mism" not in st.session_state or "gmv_mism" not in st.session_state:
    st.info("No validations found. Run **Data Validation** first.")
    st.stop()

cogs_mism = st.session_state.cogs_mism
ddp_tax_mism = st.session_state.ddp_tax_mism
gmv_mism = st.session_state.gmv_mism

# Tolerances (defaults if missing)
tols = st.session_state.get("tols", {"atol": 0.01, "rtol": 0.01, "coexist_tol": 0.01})
atol = tols.get("atol", 0.01)
rtol = tols.get("rtol", 0.01)
coexist_tol = tols.get("coexist_tol", 0.01)

# Build tiny retrieval over embedded rules/specs (no external data)
rules_md = f"""
# Validation Rules
- **COGS rule**: COGS ~= GMV Net VAT - TLG Fee - DDP Services, within atol={atol} or rtol={rtol}.
- **DDP/%Tax coexistence**: Flag rows where both `% Tax` and `DDP Services` > {coexist_tol}.
- **GMV rule**: GMV EUR ~= Sell Price - Discount - DDP Services, within atol={atol} or rtol={rtol}.

# Columns
Required columns include: COGS, GMV Net VAT, TLG Fee, DDP Services, % Tax, Sell Price, Discount, GMV EUR.
"""
rules_docs = [Document(text=rules_md, metadata={"source": "embedded_rules"})]
rules_index = VectorStoreIndex.from_documents(rules_docs)
rules_qa = rules_index.as_query_engine(similarity_top_k=3)

# Prepare SMALL samples for the agent; never give full DF
cogs_sample = cogs_mism.head(50).to_dict(orient="records")
ddp_sample = ddp_tax_mism.head(50).to_dict(orient="records")
gmv_sample = gmv_mism.head(50).to_dict(orient="records")

def get_check_summary():
    return {
        "cogs_mismatches": int(len(cogs_mism)),
        "ddp_tax_mismatches": int(len(ddp_tax_mism)),
        "gmv_mismatches": int(len(gmv_mism)),
        "cogs_sample": cogs_sample,
        "ddp_sample": ddp_sample,
        "gmv_sample": gmv_sample,
    }

summary_tool = FunctionTool.from_defaults(
    fn=get_check_summary,
    name="get_check_summary",
    description=(
        "Return mismatch counts and small samples for COGS, DDP/%Tax, and GMV checks. "
        "This tool is read-only and exposes only small samples."
    ),
)

rules_tool = QueryEngineTool.from_defaults(
    rules_qa,
    name="rules_qa",
    description="Q&A over the embedded validation rules and required columns.",
)

agent = ReActAgent.from_tools(
    tools=[summary_tool, rules_tool],
    verbose=False,
    system_prompt=(
        "You assist with invoicing sanity checks. Use tools for facts. "
        "Never claim to access databases or run SQL. You only see the provided samples."
    ),
)

st.info(
    "Security note: the assistant has **no access** to BigQuery or your full data. "
    "It only reads small, in-memory samples and an embedded rules sheet."
)

user_q = st.text_input("Ask the assistant (e.g., 'Summarize top issues').")
if st.button("Ask"):
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                resp = agent.chat(user_q)
                st.markdown(str(resp))
            except Exception as e:
                st.error(f"Assistant error: {e}")
