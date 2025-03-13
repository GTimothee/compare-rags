from datasets import load_dataset
from llama_index.core import PropertyGraphIndex
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.mistral import Mistral
# from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_openai import OpenAIEmbeddings
import datasets
import time
from pathlib import Path
from llama_index.core import Document

load_dotenv()


"""# Documentation

pip install llama-index
pip install llama-index-llms-openai-like llama-index-embeddings-openai

Extracted from the docs.

## data extraction
2 ways of extracting data: 
-  Schema-Guided Extraction: Define allowed entity types, relationship types, and their connections in a schema. The LLM will only extract graph data that conforms to this schema.
- Free-Form Extraction: Let the LLM infer the entities, relationship types and schema directly from your data in a free-form manner.
- (there is a third one but it is when nodes already contain the relationships in their properties, so the graph has already been constructed you are just loading it)
- "Mix and match these extraction approaches for fine-grained control over your graph structure."

How to query:
1. LLMSynonymRetriever: Expand your query into relevant keywords and synonyms and find matching nodes.
2. VectorContextRetriever: Use vector similarity to find nodes that are similar to your query.
3. Cypher Queries: Call with Cypher queries to find nodes that match your query.
    TextToCypherRetriever: use llm to generate the cypher queries
4. Create custom retrievers

Retrievers can be combined:

```python
from llama_index.indices.property_graph import VectorContextRetriever, LLMSynonymRetriever

vector_retriever = VectorContextRetriever(index.property_graph_store, embed_model=embed_model)  
synonym_retriever = LLMSynonymRetriever(index.property_graph_store, llm=llm)

retriever = index.as_retriever(sub_retrievers=[vector_retriever, synonym_retriever])
```
"""

llm = OpenAILike(
        model="Llama-3-70B-Instruct", 
        is_chat_model=True, 
        is_function_calling_model=False,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"))

embed_model = LangchainEmbedding(
        langchain_embeddings=OpenAIEmbeddings(
            # dimensions=1024,
            model="gte-large-en-v1.5",
            api_key=os.getenv("embedding_OPENAI_API_KEY"),
            base_url=os.getenv("embedding_OPENAI_BASE_URL")
            # timeout=500,
        )
    )


if __name__ == "__main__":
    
    index_dirpath = Path('data/llama_index')
    index_dirpath.mkdir()

    # corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")["train"]
    # docs = corpus["document"]
    docs = datasets.load_dataset("m-ric/huggingface_doc", split="train").select(range(8))
    docs = [Document(text=sample['text']) for sample in docs]
    
    # # mistral llm x sentence transformers emb
    # llm = Mistral(model="mistral-large-latest", temperature=0.3)
    # sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    # embed_model = HuggingFaceEmbedding(model=sentence_transformer_model)

    
    kg_extractor = SimpleLLMPathExtractor(llm=llm)

    t = time.time()
    index = PropertyGraphIndex.from_documents(
        docs, 
        kg_extractors=[kg_extractor], 
        embed_model=embed_model,
        show_progress=True)
    ingestion_time = time.time() - t
    print(f'Ingestion time: {ingestion_time:.2f} seconds.')

    print('Saving the index to disk...')
    index.storage_context.persist(persist_dir=str(index_dirpath))
    print('Success')
    