from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Подключение к Qdrant
    qdrant_host: str = Field("localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field("documents", alias="QDRANT_COLLECTION")

    # Модель эмбеддингов
    embedding_model: str = Field("Octen/Octen-Embedding-0.6B", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(1024, alias="EMBEDDING_DIM")

    # Инференс (генеративная модель)
    generator_model: str = Field("QuantTrio/Qwen3.5-9B-AWQ", alias="GENERATOR_MODEL")
    generator_base_url: str = Field("http://localhost:8001/v1", alias="GENERATOR_BASE_URL")
    generator_api_key: str = Field("local-vllm-key", alias="GENERATOR_API_KEY")
    generator_timeout: int = Field(900, alias="GENERATOR_TIMEOUT")
    system_prompt: str = Field("", alias="SYSTEM_PROMPT")
    max_new_tokens: int = Field(384, alias="MAX_NEW_TOKENS")
    temperature: float = Field(0.2, alias="TEMPERATURE")
    torch_dtype: str = Field("float16", alias="TORCH_DTYPE")

    # Поиск
    top_k: int = Field(5, alias="TOP_K")
    context_window_turns: int = Field(3, alias="CONTEXT_WINDOW_TURNS")

    # Чанкинг
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")

    # RAGAS оценка
    ragas_config_path: str = Field("eval/ragas_config.json", alias="RAGAS_CONFIG_PATH")
    ragas_testset_path: str = Field("eval/testset.jsonl", alias="RAGAS_TESTSET_PATH")
    ragas_output_dir: str = Field("eval/results", alias="RAGAS_OUTPUT_DIR")
    ragas_reset_index: bool = Field(False, alias="RAGAS_RESET_INDEX")
    ragas_timeout: int = Field(900, alias="RAGAS_TIMEOUT")
    ragas_max_workers: int = Field(1, alias="RAGAS_MAX_WORKERS")
    ragas_llm_model: str = Field("QuantTrio/Qwen3.5-9B-AWQ", alias="RAGAS_LLM_MODEL")
    ragas_llm_base_url: str = Field("http://localhost:8001/v1", alias="RAGAS_LLM_BASE_URL")
    ragas_llm_api_key: str = Field("local-vllm-key", alias="RAGAS_LLM_API_KEY")
    ragas_llm_temperature: float = Field(0.0, alias="RAGAS_LLM_TEMPERATURE")
    ragas_llm_max_tokens: int = Field(512, alias="RAGAS_LLM_MAX_TOKENS")
    ragas_llm_timeout: int = Field(900, alias="RAGAS_LLM_TIMEOUT")
    ragas_llm_wait_timeout: int = Field(600, alias="RAGAS_LLM_WAIT_TIMEOUT")
    ragas_llm_wait_interval: float = Field(5.0, alias="RAGAS_LLM_WAIT_INTERVAL")

    model_config = {"env_file": ".env", "populate_by_name": True, "extra": "ignore"}


settings = Settings()
