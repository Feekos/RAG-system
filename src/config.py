from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Qdrant connection
    qdrant_host: str = Field("localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field("documents", alias="QDRANT_COLLECTION")

    # Embedding model (sentence-transformers compatible)
    embedding_model: str = Field("BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(1024, alias="EMBEDDING_DIM")

    # Generation model (HuggingFace ID or local path)
    # Qwen3-4B-Instruct-2507: 4B params, ~8 GB weights in fp16/bf16.
    generator_model: str = Field("Qwen/Qwen3-4B-Instruct-2507", alias="GENERATOR_MODEL")
    max_new_tokens: int = Field(512, alias="MAX_NEW_TOKENS")
    temperature: float = Field(0.7, alias="TEMPERATURE")
    torch_dtype: str = Field("auto", alias="TORCH_DTYPE")

    # Retrieval
    top_k: int = Field(5, alias="TOP_K")

    # Chunking
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")

    # RAGAS evaluation
    ragas_config_path: str = Field("eval/ragas_config.json", alias="RAGAS_CONFIG_PATH")
    ragas_testset_path: str = Field("eval/testset.jsonl", alias="RAGAS_TESTSET_PATH")
    ragas_output_dir: str = Field("eval/results", alias="RAGAS_OUTPUT_DIR")

    model_config = {"env_file": ".env", "populate_by_name": True, "extra": "ignore"}


settings = Settings()
