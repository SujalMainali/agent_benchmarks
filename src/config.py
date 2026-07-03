import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    hf_token: str
    model_id: str
    provider: str
    max_new_tokens: int
    temperature: float
    do_sample: bool
    summary_every_n_turns: int
    recent_window_turns: int
    max_tool_steps: int


def _get_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def load_settings() -> Settings:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")

    return Settings(
        hf_token=token,
        model_id=os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        provider=os.getenv("HF_PROVIDER", "auto"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        do_sample=_get_bool("DO_SAMPLE", "false"),
        summary_every_n_turns=int(os.getenv("SUMMARY_EVERY_N_TURNS", "4")),
        recent_window_turns=int(os.getenv("RECENT_WINDOW_TURNS", "6")),
        max_tool_steps=int(os.getenv("MAX_TOOL_STEPS", "3")),
    )