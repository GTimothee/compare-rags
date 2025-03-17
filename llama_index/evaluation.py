
from datasets import load_dataset
from rag import RAG

if __name__ == "__main__":
    ds = load_dataset("GTimothee/huggingface_doc_qa_10_docs")['train']
    rag_engine = RAG()
    for sample in ds:
        answer = rag_engine.run(sample['question'])