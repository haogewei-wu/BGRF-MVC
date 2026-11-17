"""Authorized evaluation-only entry point for ground-truth labels."""

from bgrfmvc.data.loader import load_labels_for_evaluation as _load_labels


def load_evaluation_labels(dataset_config):
    """Load labels after predictions exist, for final metrics only."""
    return _load_labels(dataset_config)
