import streamlit as st
from crew import crew
# Streamlit UI
st.title("Match Grid Result Query System")
query = st.text_area("Enter your query:")

if st.button("Search"):
    if query:
        result = crew.kickoff(inputs={'query': query})
        output_text = result["raw"] if isinstance(result, dict) and "raw" in result else str(result)
        st.write(output_text)
    else:
        st.warning("Please enter a query.")






# import streamlit as st
# from crew import crew

# # Streamlit UI
# st.title("Match Grid Result Query System")

# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# query = st.text_area("Enter your query:")

# if st.button("Search"):
#     if query:
#         result = crew.kickoff(inputs={'query': query})
#         output_text = result["raw"] if isinstance(result, dict) and "raw" in result else str(result)

#         # Store query and response in chat history
#         st.session_state.chat_history.append({"query": query, "response": output_text})

# # Display chat history
# st.subheader("Conversation History:")
# for chat in st.session_state.chat_history:
#     st.write(f"**You:** {chat['query']}")
#     st.write(f"**System:** {chat['response']}")
#     st.write("---")




