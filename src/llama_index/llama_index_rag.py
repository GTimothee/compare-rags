import os
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from src.configuration import Config
from src.llama_index.models import get_llm, get_embedding_model


class LlamaIndexRag:
    def __init__(self, config: Config):
        llm = get_llm(config.llm)
        embed_model = get_embedding_model(config.embedding_model)

        if config.graph_store == 'neo4j':
            graph_store = Neo4jPropertyGraphStore(
                username=os.getenv('NEO4J_USERNAME'),
                password=os.getenv('NEO4J_PASSWORD'),
                url=os.getenv('NEO4J_URL'),
                database=os.getenv('NEO4J_DATABASE'),
            )
            index = PropertyGraphIndex.from_existing(
                property_graph_store=graph_store,
                # vector_store=vector_store,
                embed_kg_nodes=True,
            )
        else:
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=config.index_dirpath)
            )

        self.query_engine = index.as_query_engine(
            llm=llm,
            embed_model=embed_model,
            include_text=True,  # Whether to include source-text in the retriever results.
            # you can add more kwargs relative to the retrievers you use
            # To customize synonym retriever, see also https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/property_graph/sub_retrievers/llm_synonym.py#L30
            # To customize vector retriever, see also https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/property_graph/sub_retrievers/vector.py#L22
        ) 

    def run(self, question: str):
        response = self.query_engine.query(question)
        source_nodes = response.source_nodes
        data = []
        for node_with_score in source_nodes:
            data.append({
                'score': node_with_score.score,
                'node_text': node_with_score.node.get_text()
            })
        return {
            "text": response.response,
            'context': data
        }