import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    framework: str
    description: str
    dataset_path: str
    eval_dataset_path: str
    index_dirpath: str
    n_samples: int = Field(default=1)
    output_dir: str
    llm: str
    eval_llm: str
    embedding_model: str
    version: str
    graph_store: str = Field(default=None)


def load_config(config_filepath: str) -> Config:
    with open(config_filepath, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)
