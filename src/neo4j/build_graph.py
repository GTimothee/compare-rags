import os
import asyncio
import datasets

from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import MistralAILLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()


if __name__ == "__main__":

    
    llm = MistralAILLM(
        model_name="mistral-large-latest",
        api_key=os.getenv('MISTRAL_API_KEY'),
    )

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_DB_URL"), 
        auth=(os.getenv("NEO4J_DB_USERNAME"), os.getenv("NEO4J_DB_PWD")), 
        database=os.getenv('NEO4J_DB_NAME')
    )

    kg_builder = SimpleKGPipeline(
        llm=llm, 
        driver=driver, 
        embedder=SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2"), 
        from_pdf=False, 
    ) 

    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train").select(range(30))
    processed_chunks_file = "processed_chunks.txt"

    def load_processed_chunks():
        if os.path.exists(processed_chunks_file):
            with open(processed_chunks_file, "r") as file:
                return set(int(line.strip()) for line in file)
        return set()

    def save_processed_chunk(index):
        with open(processed_chunks_file, "a") as file:
            file.write(f"{index}\n")

    async def main():
        processed_chunks = load_processed_chunks()
        for index, doc in enumerate(tqdm(ds, desc="Processing documents")):
            if index not in processed_chunks:
                await kg_builder.run_async(text=doc["text"])
                save_processed_chunk(index)

    asyncio.run(main())