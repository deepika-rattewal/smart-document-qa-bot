from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class QAEngine:
    
    def __init__(self, api_key: str, persist_directory: str = "data/chroma_db"):
       
        self.api_key = api_key
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vector_store = None
        self.qa_chain = None
    
    def ingest_documents(self, texts: List[str], metadata: dict = None) -> None:

        try:
            logger.info(f"Ingesting {len(texts)} text chunks into vector store")
            
            # Convert strings to Document objects (LangChain requirement)
            documents = [
                Document(
                    page_content=text,
                    metadata=metadata or {"source": "uploaded_document"}
                )
                for text in texts
            ]
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            self.vector_store.persist()
            
            logger.info("Documents ingested successfully")
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            raise
    
    def initialize_qa_chain(self) -> None:
        if self.vector_store is None:
            raise ValueError("No documents ingested. Call ingest_documents() first.")
        
        try:
            llm = OpenAI(
                temperature=0.7,
                openai_api_key=self.api_key,
                model_name="gpt-3.5-turbo"
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3} 
                ),
                return_source_documents=True
            )
            
            logger.info("Q&A chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Q&A chain: {str(e)}")
            raise
    
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
       
        if self.qa_chain is None:
            raise ValueError("Q&A chain not initialized. Call initialize_qa_chain() first.")
        
        try:
            logger.info(f"Answering question: {question}")
            
            result = self.qa_chain({"query": question})
            
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            sources = [doc.page_content[:200] + "..." for doc in source_docs]
            
            logger.info(f"Answer generated with {len(sources)} sources")
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise