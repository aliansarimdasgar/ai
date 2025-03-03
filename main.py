import streamlit as st
from crew import crew
from tools import load_or_build_vector_store

# Load vector store
vector_store = load_or_build_vector_store()

# Streamlit UI
st.title("🤖 MatchGrid AI - Reconciliation Analysis Chatbot")
st.markdown("Identify mismatches, duplicates, and missing records efficiently!")

user_input = st.text_input("Enter your query:")

if st.button("Analyze"):
    if user_input:
        result = crew.kickoff(inputs={"query": user_input})
        st.success("📊 AI Response: " + result)
    else:
        st.warning("Please enter a query!")
