import argparse
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def push_dataset_to_hub(json_filepath, dataset_name, repo_id, token):
    # Load the JSON file
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    # Create a Dataset from the JSON data
    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"train": dataset})
    
    # Push the dataset to the Hugging Face Hub
    api = HfApi()
    api.upload_dataset(
        dataset_dict,
        repo_id=repo_id,
        token=token,
        dataset_name=dataset_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a new dataset to the Hugging Face Hub.")
    parser.add_argument("json_filepath", type=str, help="Path to the input JSON file.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument("repo_id", type=str, help="Repository ID on the Hugging Face Hub.")
    parser.add_argument("token", type=str, help="Hugging Face Hub authentication token.")
    
    args = parser.parse_args()
    
    push_dataset_to_hub(args.json_filepath, args.dataset_name, args.repo_id, args.token)