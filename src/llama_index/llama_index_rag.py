from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage


class LlamaIndexRag:
    def __init__(self, index_dirpath, llm, embed_model):
        Settings.llm = llm
        Settings.embed_model = embed_model

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