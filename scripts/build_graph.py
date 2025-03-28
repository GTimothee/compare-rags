from pathlib import Path
import argparse
import time
import yaml
from dotenv import load_dotenv

import datasets
from src.configuration import load_config
from src.llama_index.graph_builder import LlamaIndexGraphBuilder
from src.lightrag.rag import LightRag

parser = argparse.ArgumentParser(description='Build a property graph index.')
parser.add_argument('config', type=str, help='Path to the config file.')
args = parser.parse_args()


load_dotenv()


if __name__ == "__main__":
    config = load_config(args.config)
    ds = datasets.load_dataset(config.dataset_path, split="train").select(range(config.n_samples))

    t = time.time()
    if config.framework == 'llama_index':
        LlamaIndexGraphBuilder(config).build(ds)
    elif config.framework == 'lightrag':
        LightRag(config).build(ds)
    else:
        raise NotImplementedError(f"Unsupported framework: {config.framework}")
    ingestion_time = time.time() - t
    print(f'Ingestion time: {ingestion_time:.2f} seconds.')
    print('Success')
    