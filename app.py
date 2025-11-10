import streamlit as st
from rag.rag import RAGSystem
from langsmith import traceable

# Function to initialize the RAG system with a loading message
@traceable
def initialize_rag():
    try:
        with st.spinner('Initializing RAG system, please wait.......'):
            rag_system = RAGSystem()
        return rag_system
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

if 'rag_system' not in st.session_state:
    st.session_state['rag_system'] = initialize_rag()
    if st.session_state['rag_system'] is None:
        st.stop()

st.title("CNU Undergraduate Advisor Chatbot")
st.write("This chatbot can answer questions about CNU's undergraduate programs, courses, and policies.")

use_reranker = st.checkbox("Use Reranker?", help="Enabling this may improve answer relevance but might increase response time.")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a query before submitting.")
    else:
        with st.spinner('Generating response, please wait...'):
            rag = st.session_state['rag_system']
            response, docs = rag.answer_with_rag(query, use_reranker=use_reranker)
        
        st.subheader("Response:")
        st.write(response)
        
        st.subheader("Relevant Documents:")
        for i, doc in enumerate(docs, 1):
            st.write(f"{i}. {doc}")