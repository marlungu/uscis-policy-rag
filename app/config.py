from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file =".env", 
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    #AWS
    aws_region: str = "us-east-1"
    aws_access_key_id: str
    aws_secret_access_key: str

    #S3
    s3_bucket_name: str
    s3_prefix: str = "docs/"

    #Database
    postgres_url: str

    #Models
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    chat_model_id: str = "us.anthropic.claude-sonnet-4-6"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    search_type: str = "mmr"

    # Vector store
    collection_name: str = "uscis_policy_documents"

    @computed_field
    @property
    def s3_uri(self) -> str:
        return f"s3://{self.s3_bucket_name}/{self.s3_prefix}"

settings = Settings()