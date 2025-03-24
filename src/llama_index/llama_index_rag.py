from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from src.llama_index.models import llms, embedding_models


class LlamaIndexRag:
    def __init__(self, index_dirpath, llm_name, embed_model_name):
        Settings.llm = llms[llm_name]
        Settings.embed_model = embedding_models[embed_model_name]

        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_dirpath)
        )
        self.query_engine = index.as_query_engine(
            # include_text=True
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