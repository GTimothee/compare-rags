import os

from neo4j import GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import MistralAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

from dotenv import load_dotenv

load_dotenv()


class RAG:

    supported_llm_types = ["mistral"]

    def __init__(self, driver, index_name: str, llm_type: str, llm_api_key: str, top_k: int = 5):

        self.llm_type = llm_type

        if not llm_type in self.supported_llm_types:
            raise ValueError(f"llm_type must be one of {self.supported_llm_types}")

        if llm_type == "mistral":
            self.llm = MistralAILLM(
                model_name="mistral-small-latest",
                api_key=llm_api_key,
            )
        
        self.retriever_config = {"top_k": top_k}

        self.driver = driver
        retriever = VectorRetriever(driver, index_name, SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2"))
        self.rag = GraphRAG(retriever=retriever, llm=self.llm)
            
    def run_query(self, query):
        response = self.rag.search(query_text=query, retriever_config=self.retriever_config)
        print(response.answer)
        return response
    

if __name__ == "__main__":

    index_name = "test-index"
    DIMENSION=1536

    print(f'os.getenv(NEO4J_DB_URL)={os.getenv("NEO4J_DB_URL")}')

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_DB_URL"), 
        auth=(os.getenv("NEO4J_DB_USERNAME"), os.getenv("NEO4J_DB_PWD")), 
        database=os.getenv('NEO4J_DB_NAME')
    )

    create_vector_index(
        driver,
        index_name,
        label="Document",
        embedding_property="vectorProperty",
        dimensions=DIMENSION,
        similarity_fn="euclidean",
    )

    rag = RAG(
        driver, 
        index_name=index_name, 
        llm_type="mistral", 
        llm_api_key=os.getenv('MISTRAL_API_KEY')
    )
    
    answer = rag.run_query("What is the capital of France?")
    print(answer)