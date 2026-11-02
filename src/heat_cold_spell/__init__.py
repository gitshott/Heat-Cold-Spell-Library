"""Public API for the heat and cold spell library."""

from .heat_cold_spell import (
    analyze_history,
    compare_models,
    predict_future,
    predict_spells,
    prepare_data,
    train_models,
)

__all__ = [
    "prepare_data",
    "analyze_history",
    "compare_models",
    "train_models",
    "predict_future",
    "predict_spells",
]