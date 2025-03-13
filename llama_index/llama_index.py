from datasets import load_dataset
from llama_index.core import PropertyGraphIndex
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistral import Mistral

load_dotenv()


"""# Documentation

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


if __name__ == "__main__":
    
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")["train"]
    docs = corpus["document"]
    
    llm = Mistral(model="mistral-large-latest", temperature=0.3)
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_model = HuggingFaceEmbedding(model=sentence_transformer_model)

    # kg_extractor = SimpleLLMPathExtractor(llm=llm)
    # index = PropertyGraphIndex.from_documents(
    #     docs, 
    #     kg_extractors=[kg_extractor], 
    #     embed_model=embed_model,,
    #     show_progress=True)
    
    # index.property_graph.save_graph_to_html("knowledge_graph.html")


    ## >>> evaluation
    # evaluation_set = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
    # query_engine_KnowledgeGraph = index.as_query_engine(
    #     include_text=True
    # ) 
    