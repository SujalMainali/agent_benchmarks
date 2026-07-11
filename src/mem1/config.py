"""MEM1 configuration settings."""

from dataclasses import dataclass, field
from os import getenv
from dotenv import load_dotenv


@dataclass
class Mem1Settings:
    """Configuration for MEM1 agent."""
    
    # Model settings
    model_id: str = "Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release"
    hf_token: str = ""
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Context settings
    max_context_chars: int = 2000
    max_think_chars: int = 1000
    max_information_chars: int = 800
    
    # Reasoning loop settings
    max_reasoning_steps: int = 10
    
    # Retrieval settings
    retriever_type: str = "web"  # "web", "corpus", "none"
    serper_api_key: str = ""
    top_k_results: int = 3


def load_mem1_settings() -> Mem1Settings:
    """Load MEM1 settings from environment variables."""
    load_dotenv()
    
    return Mem1Settings(
        model_id=getenv("MEM1_MODEL_ID", "Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release"),
        hf_token=getenv("MEM1_HF_TOKEN", getenv("HUGGINGFACEHUB_API_TOKEN", "")),
        max_new_tokens=int(getenv("MEM1_MAX_NEW_TOKENS", "512")),
        temperature=float(getenv("MEM1_TEMPERATURE", "0.7")),
        top_p=float(getenv("MEM1_TOP_P", "0.9")),
        max_context_chars=int(getenv("MEM1_MAX_CONTEXT_CHARS", "2000")),
        max_think_chars=int(getenv("MEM1_MAX_THINK_CHARS", "1000")),
        max_information_chars=int(getenv("MEM1_MAX_INFORMATION_CHARS", "800")),
        max_reasoning_steps=int(getenv("MEM1_MAX_REASONING_STEPS", "10")),
        retriever_type=getenv("MEM1_RETRIEVER_TYPE", "web"),
        serper_api_key=getenv("SERPER_API_KEY", ""),
        top_k_results=int(getenv("MEM1_TOP_K_RESULTS", "3")),
    )