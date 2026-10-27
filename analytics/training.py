"""Summaries of NeuralStore's bounded persisted training-loss history."""

from __future__ import annotations

from statistics import mean, median
from typing import Any

from neuralstore import NeuralStore


def collect_training_history(store: NeuralStore, last: int = 100) -> dict[str, Any]:
    """Summarize the most recent combined training losses."""
    if last < 1:
        raise ValueError("last must be >= 1")
    history = list(store.loss_history[-last:])
    midpoint = len(history) // 2
    first_half = history[:midpoint]
    second_half = history[midpoint:]
    return {
        "training_steps": store.training_steps,
        "history_limit": store.MAX_LOSS_HISTORY,
        "available_points": len(store.loss_history),
        "points_reported": len(history),
        "last_loss": store.last_loss,
        "mean": mean(history) if history else None,
        "median": median(history) if history else None,
        "first_half_mean": mean(first_half) if first_half else None,
        "second_half_mean": mean(second_half) if second_half else None,
        "losses": history,
    }


def print_training_history(store: NeuralStore, last: int = 100) -> None:
    """Print recent loss metrics with a careful interpretation note."""
    report = collect_training_history(store, last=last)
    print("Training history (combined loss)")
    print(f"Training steps total: {report['training_steps']}")
    print(f"History points available: {report['available_points']} "
          f"(rolling limit: {report['history_limit']})")
    print(f"Points reported: {report['points_reported']}")
    if not report["losses"]:
        print("No training-loss history is available yet.")
        print("Loss history begins accumulating after this version's training steps.")
    else:
        print(f"Latest loss: {report['last_loss']:.6f}")
        print(f"Recent mean: {report['mean']:.6f}")
        print(f"Recent median: {report['median']:.6f}")
        if report["first_half_mean"] is not None and report["second_half_mean"] is not None:
            print(f"Earlier half mean: {report['first_half_mean']:.6f}")
            print(f"Later half mean: {report['second_half_mean']:.6f}")
    print("This is training loss only; it does not establish generalization or retrieval quality.")
