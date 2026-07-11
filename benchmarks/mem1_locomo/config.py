"""Configuration for MEM1 LoCoMo benchmark."""

from dataclasses import dataclass
from os import getenv
from dotenv import load_dotenv

from src.mem1.config import Mem1Settings, load_mem1_settings


@dataclass
class Mem1LoCoMoSettings:
    """Combined settings for MEM1 + LoCoMo benchmark."""
    
    # MEM1 settings
    mem1: Mem1Settings = None
    
    # LoCoMo data settings
    data_file: str = "data/locomo/demo.jsonl"
    run_mode: str = "single"  # "single" or "batch"
    sample_id: str = ""  # For single mode
    max_samples: int = -1  # -1 for all
    
    # Output settings
    output_dir: str = "results/mem1_locomo"
    save_trajectories: bool = True
    save_think_history: bool = True
    
    def __post_init__(self):
        if self.mem1 is None:
            self.mem1 = load_mem1_settings()


def load_mem1_locomo_settings() -> Mem1LoCoMoSettings:
    """Load MEM1 LoCoMo settings from environment."""
    load_dotenv()
    
    return Mem1LoCoMoSettings(
        mem1=load_mem1_settings(),
        data_file=getenv("MEM1_LOCOMO_DATA_FILE", getenv("LOCOMO_DATA_FILE", "data/locomo/demo.jsonl")),
        run_mode=getenv("MEM1_LOCOMO_RUN_MODE", getenv("LOCOMO_RUN_MODE", "single")),
        sample_id=getenv("MEM1_LOCOMO_SAMPLE_ID", getenv("LOCOMO_SAMPLE_ID", "")),
        max_samples=int(getenv("MEM1_LOCOMO_MAX_SAMPLES", "-1")),
        output_dir=getenv("MEM1_LOCOMO_OUTPUT_DIR", "results/mem1_locomo"),
        save_trajectories=getenv("MEM1_LOCOMO_SAVE_TRAJECTORIES", "true").lower() == "true",
        save_think_history=getenv("MEM1_LOCOMO_SAVE_THINK_HISTORY", "true").lower() == "true",
    )