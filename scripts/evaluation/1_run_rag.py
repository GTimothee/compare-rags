
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import os
import argparse
import json

from src.configuration import load_config
from src.lightrag.rag import LightRag
from src.llama_index.llama_index_rag import LlamaIndexRag


parser = argparse.ArgumentParser(description='Evaluate a Q/A system using llm as judge.')
parser.add_argument('config', type=str, help='The path to the configuration YAML file.')
parser.add_argument('--limit', type=int, default=-1)
args = parser.parse_args()

load_dotenv()


if __name__ == "__main__":
    
    config = load_config(args.config)

    logging.basicConfig(
        filename=f"{config.framework}_answers.log",
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    ds = load_dataset(config.eval_dataset_path)['train']

    # Load RAG engine
    if config.framework == "llama_index":
        rag_engine = LlamaIndexRag(config)
    elif config.framework == "lightrag":
        rag_engine = LightRag(config.index_dirpath)
    else: 
        raise NotImplementedError(f"Unsupported framework: {config.framework}")

    # Generation loop
    results = []
    for sample_idx, sample in tqdm(enumerate(ds), total=len(ds), desc=f"RAG running on dataset {config.eval_dataset_path}..."):
        if sample_idx == args.limit:
            logging.info(f"Limit of {args.limit} samples reached. Stopping generation.")
            break
        logging.info(f"Processing sample {sample_idx}...")

        try:
            logging.info("- Running RAG engine...")
            rag_answer = rag_engine.run(sample['question'])
            logging.info(f"- RAG response: {rag_answer}")
            sample['rag_answer'] = rag_answer['text']
            sample['rag_context'] = rag_answer['context']
        except Exception as e:
            logging.error(f"Error processing sample {sample_idx}: {e}")
            sample['rag_answer'] = None
            sample['rag_context'] = None
            results.append(sample)
            continue

        
        results.append({
            'sample_idx': sample_idx,
            'question': sample['question'],
            'answer': sample['answer'],
            'rag_answer': sample['rag_answer'],
            'rag_context': sample['rag_context']
        })

    print('Saving results...')
    output_path = os.path.join(config.output_dir, 'answers.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)