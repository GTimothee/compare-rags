from pathlib import Path
import argparse
import time
import yaml
from dotenv import load_dotenv

import datasets
from src.llama_index.graph_builder import LlamaIndexGraphBuilder
from src.lightrag.rag import LightRag

parser = argparse.ArgumentParser(description='Build a property graph index.')
parser.add_argument('config', type=str, help='Path to the config file.')
args = parser.parse_args()


load_dotenv()


if __name__ == "__main__":
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    index_dirpath = config['index_dirpath']
    dataset_path = config['dataset_path']
    framework = config['framework']

    ds = datasets.load_dataset(dataset_path, split="train").select(range(8))

    t = time.time()
    if framework == 'llama_index':
        LlamaIndexGraphBuilder(index_dirpath, config["llm"], config["embedding_model"]).build(ds)
    elif framework == 'lightrag':
        LightRag(index_dirpath).build(ds)
    else:
        raise NotImplementedError(f"Unsupported framework: {framework}")
    ingestion_time = time.time() - t
    print(f'Ingestion time: {ingestion_time:.2f} seconds.')
    print('Success')
    