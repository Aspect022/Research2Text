import numpy as np
import torch

# Optional imports – provide graceful fall‑backs if the libraries are not installed
try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:  # pragma: no cover
    LimeTabularExplainer = None


def compute_fairness_metrics(preds: np.ndarray, labels: np.ndarray, protected: np.ndarray):
    """Calculate demographic parity and equal opportunity differences.

    Args:
        preds: Predicted probabilities.
        labels: Ground‑truth binary labels.
        protected: Binary protected attribute (e.g., gender).
    Returns:
        Tuple (demographic_parity_diff, equal_opportunity_diff).
    """
    preds_binary = (preds >= 0.5).astype(int)
    # Demographic parity difference
    pos_rate_prot = preds_binary[protected == 1].mean()
    pos_rate_unprot = preds_binary[protected == 0].mean()
    dp_diff = abs(pos_rate_prot - pos_rate_unprot)

    # Equal opportunity (true positive rate) difference
    tp_prot = ((preds_binary == 1) & (labels == 1) & (protected == 1)).sum()
    fn_prot = ((preds_binary == 0) & (labels == 1) & (protected == 1)).sum()
    tpr_prot = tp_prot / (tp_prot + fn_prot + 1e-8)

    tp_unprot = ((preds_binary == 1) & (labels == 1) & (protected == 0)).sum()
    fn_unprot = ((preds_binary == 0) & (labels == 1) & (protected == 0)).sum()
    tpr_unprot = tp_unprot / (tp_unprot + fn_unprot + 1e-8)

    eo_diff = abs(tpr_prot - tpr_unprot)
    return dp_diff, eo_diff


def get_shap_values(model: torch.nn.Module, background_data: np.ndarray, instance: np.ndarray):
    """Return SHAP values for a single instance.

    If the *shap* library is unavailable, a zero‑vector of the appropriate shape
    is returned so that downstream code can continue to run.
    """
    if shap is None:
        # Fallback: return zeros matching the feature dimension
        return np.zeros(instance.shape[1]).tolist()

    # KernelExplainer expects a callable returning model outputs as numpy
    def model_wrapper(x):
        with torch.no_grad():
            # Ensure the tensor is on the same device as the model
            return model(torch.from_numpy(x).float()).cpu().numpy().squeeze()

    # Pick a small subset of background data for efficiency
    explainer = shap.KernelExplainer(model_wrapper, background_data[:100])
    shap_vals = explainer.shap_values(instance)
    # ``shap_vals`` can be a list (for multi‑output models) or an array; we standardise it
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return shap_vals.tolist()


def get_lime_explanation(model: torch.nn.Module, train_data: np.ndarray, instance: np.ndarray):
    """Return LIME explanation for a single instance.

    If *lime* is not installed, a simple placeholder list is returned.
    """
    if LimeTabularExplainer is None:
        # Fallback placeholder – list of (feature, contribution) with zero contribution
        feature_names = [f"f{i}" for i in range(train_data.shape[1])]
        return [(name, 0.0) for name in feature_names]

    feature_names = [f"f{i}" for i in range(train_data.shape[1])]
    explainer = LimeTabularExplainer(
        train_data,
        feature_names=feature_names,
        class_names=["reject", "accept"],
        mode="classification",
    )

    def predict_fn(x):
        with torch.no_grad():
            return model(torch.from_numpy(x).float()).cpu().numpy().squeeze()

    exp = explainer.explain_instance(instance[0], predict_fn)
    return exp.as_list()
