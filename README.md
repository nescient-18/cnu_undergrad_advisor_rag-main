# CNU Undergraduate Advisor Chatbot

This project implements a Retrieval-Augmented Generation (RAG) system to create a chatbot that can answer questions about Christopher Newport University's undergraduate programs, particularly focusing on the Department of Physics, Computer Science and Engineering (PCSE) and the Department of Mathematics.

## Setup Instructions

### Step 1: Create and Activate a Virtual Environment

Create a virtual environment to manage dependencies. Assuming conda:

```bash
conda create --name "cnu_rag" python=3.10
conda activate cnu_rag
pip install -r requirements.txt
```
### Step 2: Run the Streamlip App

Navigate to your project directory and run:

```bash
streamlit run app.py
```

## Project Structure


The project consists of the following main components:

1. `rag.py`: Contains the `RAGSystem` class, which implements the core RAG functionality.
2. `app.py`: Implements the Streamlit web application for the chatbot interface.
3. `data/pcse.md`: Contains the knowledge base for the Department of Physics, Computer Science and Engineering.
4. `data/prompt.txt`: Contains the system prompt for the RAG system (not included in the provided files).

## Features

- Uses FAISS for efficient similarity search
- Employs HuggingFace's BGE embeddings for document representation
- Utilizes Meta's LLaMA 3 (8B parameters) as the base language model
- Optional reranking using ColBERT v2.0
- Streamlit-based user interface

## System Components

### RAG System (`rag.py`)

The `RAGSystem` class in `rag.py` implements the following key functionalities:

1. Document loading and processing
2. Vector database creation using FAISS
3. Language model loading (LLaMA 3.1 Instruct)
4. Prompt template definition
5. Optional reranking using ColBERT v2.0

The system uses a retrieval-then-read approach, where relevant documents are first retrieved based on the user's query, and then fed into the language model to generate a response.

### User Interface (`app.py`)

The Streamlit app provides a user-friendly interface for interacting with the CNU Undergraduate Advisor Chatbot. Key features include:

1. **Query Input**: Users can enter their questions about CNU's undergraduate programs, courses, and policies in a text input field.

2. **Reranker Toggle**: A checkbox allows users to enable or disable the document reranking feature. This option can potentially improve answer relevance at the cost of increased response time.

3. **Submit Button**: Triggers the RAG system to process the query and generate a response.

4. **Response Display**: The generated answer is displayed prominently for easy reading.

5. **Relevant Documents**: The system shows the top relevant documents used to construct the answer, providing transparency and additional context.

6. **Loading Indicators**: Spinner animations are displayed during system initialization and response generation to provide visual feedback on processing status.

7. **Error Handling**: The app includes error messages for scenarios such as initialization failures or empty queries.

## Customization

The RAG system is designed to be flexible and customizable. Here are the key components you can modify to adapt the system to different use cases or improve performance:

1. **Language Model**:
   - Modify `self.llm_name` in `rag.py`
   - Current: "meta-llama/Meta-Llama-3.1-8B-Instruct"
   - Consider experimenting with different model sizes or alternative models like GPT-J or BLOOM

2. **Embedding Model**:
   - Adjust `self.embedding_model_name` in `rag.py`
   - Current: "BAAI/bge-large-en-v1.5"
   - Other options include models from the sentence-transformers family or domain-specific embeddings

3. **Reranker Model**:
   - Change `self.reranker_model` in `rag.py`
   - Current: "colbert-ir/colbertv2.0"
   - Alternatives include monoBERT or other ColBERT versions

4. **System Prompt**:
   - Edit the content of `data/prompt.txt`
   - Customize the prompt to better suit the specific domain or to improve the chatbot's persona and response style

5. **Knowledge Base**:
   - Update or expand `data/pcse.md`
   - Add new markdown files for additional departments or topics
   - Modify the `_load_and_process_docs` method in `rag.py` to handle multiple knowledge base files

6. **Retrieval Parameters**:
   - Adjust `num_retrieved_docs` and `num_docs_final` in the `answer_with_rag` method
   - Fine-tune these values to balance between comprehensive context and focused responses

7. **Vector Store**:
   - The system currently uses FAISS, but you could experiment with other vector stores like Pinecone or Weaviate by modifying the `_load_vectors` method

8. **UI Customization**:
   - Modify `app.py` to add new Streamlit components or change the layout
   - Consider adding features like conversation history or topic filtering

When making customizations, especially to core components like the LLM or embedding model, be sure to test thoroughly to ensure compatibility and performance improvements.