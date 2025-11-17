import streamlit as st

# Use Streamlit's navigation API to group pages
# Ref: https://docs.streamlit.io/develop/api-reference/navigation/st.navigation

pages = {
    "Invoice Checker": [
        st.Page("1_File_Upload.py", title="File Upload", icon="📥"),
        st.Page("pages/2_Data_Validation.py", title="File Data Validation", icon="📄"),
        st.Page("pages/4_BI_Cross_Check.py", title="BQ Cross Check", icon="📊"),
        st.Page("pages/5_Export_Data.py", title="Export Updated Data", icon="📤"),
    ],
    "Configuration": [
        st.Page("pages/6_Brand_Fee_Config.py", title="Brand Fee Configuration", icon="⚙️"),
        st.Page("pages/7_DDP_Config.py", title="DDP Duties Configuration", icon="⚙️"),
    ],
}

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()


