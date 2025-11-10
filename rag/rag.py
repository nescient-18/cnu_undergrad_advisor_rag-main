from langchain_core.documents import Document as LangchainDocument
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Tuple
import torch

class RAGSystem:
    def __init__(self):
        """
        Initializes the RAG object.

        Parameters:
            None

        Returns:
            None
        """
        
        # Specify LLM
        self.llm_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        # Specify embeddings
        self.embedding_model_name = "BAAI/bge-large-en-v1.5"

        # Specify reranker
        self.reranker_model = "colbert-ir/colbertv2.0"

        # Load and process documents
        self._docs_processed = self._load_and_process_docs(
            data_file_path = "./data/pcse.md"
        )

        # Load knowledge vector database
        self._knowledge_vector_database = self._load_vectors()

        # Load LLM
        self._reader_llm = self._load_llm()

        # Define RAG prompt template
        self._rag_prompt_template = self._define_prompt(
            prompt_file_path = "./data/prompt.txt"
        )

        # Load reranker
        self._reranker = self._load_reranker()

    def _load_and_process_docs(
        self,
        data_file_path: str,
    ) -> List[LangchainDocument]:
        """
        Load and process the documents.

        Args:
            data_file_path (str): The file path of the data file.

        Returns:
            List[LangchainDocument]: A list of processed documents.

        """

        def split_documents(
            knowledge_base: List[LangchainDocument], 
            chunk_size: int = 512,
        ) -> List[LangchainDocument]:
            """
            Split documents into chunks of maximum size `chunk_size` characters and return a list of documents.

            Args:
                knowledge_base (List[LangchainDocument]): The list of documents to be split.
                chunk_size (int): The maximum size of each chunk in characters. Defaults to 512.

            Returns:
                List[LangchainDocument]: A list of split documents.

            """
            
            # Initialize the text splitter
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                AutoTokenizer.from_pretrained(self.embedding_model_name),
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size / 10) + 5,
                add_start_index=True,
                strip_whitespace=True,
            )

            # Split the documents
            docs_processed = text_splitter.split_documents(knowledge_base)

            # Remove duplicates
            unique_texts = set()
            docs_processed_unique = []
            for doc in docs_processed:
                if doc.page_content not in unique_texts:
                    unique_texts.add(doc.page_content)
                    docs_processed_unique.append(doc)

            return docs_processed_unique

        try:
            loader = UnstructuredMarkdownLoader(file_path=data_file_path)
            raw_knowledge_base = loader.load()
            processed_docs = split_documents(raw_knowledge_base)
            return processed_docs
        except Exception as e:
            raise RuntimeError(f"Error processing documents: {str(e)}") from e

    def _load_vectors(self) -> FAISS:
        """
        Load document vectors into a FAISS vector store.

        This method creates an embedding model using HuggingFaceBgeEmbeddings,
        then uses it to embed the processed documents and store them in a
        FAISS vector store.

        Returns:
            FAISS: A FAISS vector store containing the embedded documents.
        """
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vector_store = FAISS.from_documents(
            documents=self._docs_processed,
            embedding=embedding_model,
            distance_strategy=DistanceStrategy.COSINE
        )

        return vector_store

    def _load_llm(self):
        """
        Load and configure a language model for text generation.

        This method sets up a quantized language model using the Bits and Bytes configuration,
        loads the model and tokenizer from a pre-trained checkpoint, and configures a text
        generation pipeline with specific parameters.

        Returns:
            pipeline: A Hugging Face pipeline object for text generation.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.llm_name, 
            quantization_config=bnb_config
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.llm_name
        )

        reader_llm = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.15,
            top_p=0.95,
            repetition_penalty=1.2,
            return_full_text=False,
            max_new_tokens=750,
        )

        return reader_llm

    def _define_prompt(self, prompt_file_path: str) -> str:
        """
        Define and format the RAG prompt template.

        This method loads a system prompt from a file, combines it with a user prompt
        template, and formats the entire prompt using the model's chat template.

        Args:
            prompt_file_path (str): The file path to the system prompt.

        Returns:
            str: The formatted RAG prompt template.
        """

        def load_prompt_from_file(file_path: str) -> str:
            """
            Load prompt content from a file.

            Args:
                file_path (str): The path to the prompt file.

            Returns:
                str: The content of the prompt file.
            """
            with open(file_path, "r") as file:
                return file.read()

        system_content = load_prompt_from_file(prompt_file_path)

        prompt_in_chat_format = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": """Context: {context}

                ---

                Here is the question you need to answer.
                
                Question: {question}""",
            },
        ]

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.llm_name
        )

        rag_prompt_template = tokenizer.apply_chat_template(
            prompt_in_chat_format, 
            tokenize=False, 
            add_generation_prompt=True
        )

        return rag_prompt_template

    def _load_reranker(self) -> RAGPretrainedModel:
        """
        Load a pre-trained reranker model.

        This method loads a pre-trained RAG (Retrieval-Augmented Generation) model
        to be used as a reranker in the retrieval process.

        Returns:
            RAGPretrainedModel: A pre-trained RAG model for reranking.
        """

        reranker = RAGPretrainedModel.from_pretrained(
            pretrained_model_name_or_path=self.reranker_model
        )

        return reranker
    
    def answer_with_rag(self,
                        question: str,
                        use_reranker: bool = False,
                        num_retrieved_docs: int = 30,
                        num_docs_final: int = 10) -> Tuple[str, List[str]]:
        """
        Generate an answer using Retrieval-Augmented Generation (RAG).

        This method retrieves relevant documents based on the question, optionally
        reranks them, and then uses a language model to generate an answer.

        Args:
            question (str): The question to be answered.
            use_reranker (bool, optional): Whether to use document reranking. Defaults to False.
            num_retrieved_docs (int, optional): Number of documents to retrieve initially. Defaults to 30.
            num_docs_final (int, optional): Number of documents to use for answer generation. Defaults to 10.

        Returns:
            Tuple[str, List[str]]: A tuple containing the generated answer and the list of relevant documents.
        """
        
        relevant_docs = self._knowledge_vector_database.similarity_search(query=question, k=num_retrieved_docs)
        relevant_docs = [doc.page_content for doc in relevant_docs]
        
        if use_reranker:
            print("Reranking documents...")
            relevant_docs = self._reranker.rerank(question, relevant_docs, k=num_docs_final)
            relevant_docs = [doc["content"] for doc in relevant_docs]
        else:
            print("Skipping reranking...")
            relevant_docs = relevant_docs[:num_docs_final]
        
        context = "\nExtracted documents:\n" + "\n".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(relevant_docs)])
        final_prompt = self._rag_prompt_template.format(question=question, context=context)
        
        answer = self._reader_llm(final_prompt)[0]["generated_text"]
        
        return answer, relevant_docs