import pandas as pd
from pydantic import BaseModel
from typing import Any, Optional
from enum import Enum


class EmbeddingConfig(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    embed_sentences: bool = True
    use_multiprocessing: bool = False


class TextProcessingConfig(BaseModel):
    min_words_len: Optional[int] = 3
    max_words_len: Optional[int] = None
    sampling: bool = True
    max_sample_size: int = 10000
    clean_digits: bool = True


class TextData(BaseModel):
    text_field: str
    text_df: pd.DataFrame
    sentence_df: pd.DataFrame
    text_processing_config: TextProcessingConfig

    class Config:
        arbitrary_types_allowed = True


class EmbeddingData(BaseModel):
    embedding_model: str
    text_embeddings: Any
    sentence_embeddings: Any
    embedding_config: EmbeddingConfig

    class Config:
        arbitrary_types_allowed = True


class SearchMethod(str, Enum):
    semantic = "semantic"
    non_semantic = "non-semantic"
    both = "both"

    @classmethod
    def values(cls):
        possible_values = [x.value for x in list(cls)]
        return possible_values
