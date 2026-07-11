import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    # Provider selection ("huggingface" | "openai").
    llm_provider: str

    # Shared model/runtime knobs (provider-agnostic).
    model_id: str
    temperature: float
    max_new_tokens: int
    do_sample: bool

    # Hugging Face specific.
    hf_token: str
    hf_inference_provider: str

    # OpenAI (or OpenAI-compatible) specific.
    api_key: str
    openai_base_url: str

    # Agent/memory behavior.
    summary_every_n_turns: int
    recent_window_turns: int
    max_tool_steps: int


def _get_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "huggingface").strip().lower()

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    # Only Hugging Face runs require an HF token. OpenAI runs must not be
    # blocked by a missing HF credential.
    if provider in {"huggingface", "hf"} and not hf_token:
        raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")

    # Resolve the model id, preferring an explicit per-provider override so both
    # backends can be configured side by side in a single .env.
    if provider in {"openai", "ollama", "openai_compatible"}:
        model_id = os.getenv("OPENAI_MODEL_ID") or os.getenv("MODEL_ID", "gpt-4o-mini")
    else:
        model_id = os.getenv("HF_MODEL_ID") or os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

    return Settings(
        llm_provider=provider,
        model_id=model_id,
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
        do_sample=_get_bool("DO_SAMPLE", "false"),
        hf_token=hf_token,
        hf_inference_provider=os.getenv("HF_PROVIDER", "auto"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        summary_every_n_turns=int(os.getenv("SUMMARY_EVERY_N_TURNS", "4")),
        recent_window_turns=int(os.getenv("RECENT_WINDOW_TURNS", "6")),
        max_tool_steps=int(os.getenv("MAX_TOOL_STEPS", "3")),
    )
