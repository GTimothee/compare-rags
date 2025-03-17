from llama_index.core import StorageContext, load_index_from_storage
from dotenv import load_dotenv
# evaluation_set = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
from llama_index.core import Settings
from build_graph import llm, embed_model

load_dotenv()


class RAG:
    def __init__(self):
        Settings.llm = llm
        Settings.embed_model = embed_model

        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir='data/llama_index')
        )
        self.query_engine = index.as_query_engine(
            # include_text=True
        ) 

    def run(self, question: str):
        return self.query_engine.query(question)


if __name__ == "__main__":

    rag_engine = RAG()

    for question in [
        "What is SQuAD?",
        "What is the default CPU configuration for a created endpoint?"
    ]:
        print('****')
        print(f"Question: {question}")
        response = rag_engine.run(question)
        print(f"Answer: {response}")