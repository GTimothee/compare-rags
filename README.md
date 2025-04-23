# RAG systems (classical - graph - agentic) evaluation

Systems being compared
- [ ] Classical RAG system
- [ ] graph RAG
   - [x] Llama-index implementation
   - [x] LightRAG implementation
   - [ ] Neo4j-graphrag

## Results

### huggingface 10 docs evaluation

On the ["huggingface 10 docs"](https://huggingface.co/datasets/GTimothee/huggingface_doc_qa_10_docs) evaluation dataset:

| Framework | Config | Version description | LLM | Normalized accuracy |
| --- | --- | --- | --- | --- |
| llama-index | experiments/llama_index/out-of-the-box | Llama-3-70B-Instruct | Does not use embeddings (default behaviour without using a graphDB) | 74.7% |
| llama-index | experiments/llama_index/out-of-the-box-neo4j | Mistral 7B Instruct Quantized 4 bit (AWQ) | Out of the box config + use neo4j to enable embeddings + switch llm to a 7B model | 81% |
| lightrag | experiments/lightrag/out-of-the-box | Llama-3-70B-Instruct | Default configuration, with embeddings | 81% |

## Sources

Evaluation built based on the following cookbook by huggingface: https://huggingface.co/learn/cookbook/en/rag_evaluation

