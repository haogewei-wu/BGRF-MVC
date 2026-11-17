"""Label-free preprocessing for multi-view features."""

from sklearn.preprocessing import StandardScaler


def standardize_views(X_views):
    """Fit one independent StandardScaler per view; accepts features only."""
    return [StandardScaler().fit_transform(view) for view in X_views]

