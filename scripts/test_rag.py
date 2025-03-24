"""This script enables to test your installation and graph building by running a question against the RAG engine. 
It is a simple script that takes a question as input and returns the answer from the RAG engine. 
You can run the script by executing the following command in the terminal:

```bash 
python scripts/rag.py "What is the capital of France?"
```

Note: of course ask a question that can be answered using the graph you built.
"""

from dotenv import load_dotenv
import yaml
from build_graph import llm, embed_model
from src.lightrag.rag import LightRag
from src.llama_index.llama_index_rag import LlamaIndexRag
import argparse

parser = argparse.ArgumentParser(description="Run rag with a question.")
parser.add_argument('config', type=str, required=True, help='Path to the config file.')
parser.add_argument('question', type=str, help='The question to ask the LlamaIndexRag engine.')
parser.add_argument('--verbose', action='store_true', help='Increase verbosity. At the moment it just prints the context retrieved.')
args = parser.parse_args()

load_dotenv()


if __name__ == "__main__":

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    index_dirpath = config['index_dirpath']
    framework = config['framework']

    if framework == "llama_index":
        rag_engine = LlamaIndexRag(index_dirpath, config['llm'], config['embedding_model'])
    elif framework == "lightrag":
        rag_engine = LightRag(index_dirpath)
    else: 
        raise NotImplementedError(f"Unsupported framework: {framework}")

    print(f"Question: {args.question}")
    response = rag_engine.run(args.question)
    print(f"Answer: {response['text']}")
    if args.verbose:
        print(f"Context: {response['context']}")