MODEL_ORDER = ("LinearSVC", "LDA", "RF", "DT", "XGBoost")
NETWORKS = ("WB", "DMN", "SMN", "VN")
FEATURE_MODES = ("FC", "topology", "FC+topology")
VARIANTS = ("Original", "Mask")
METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "accuracy",
    "f1",
)

MODE_FOLDERS = {
    "FC": "FC",
    "topology": "拓扑学",
    "FC+topology": "FC+拓扑学",
}
MODE_DISPLAY = {
    "FC": "FC",
    "topology": "Topology",
    "FC+topology": "FC+Topology",
}

SUBJECT_ORDER_SHA256 = (
    "a1104d7df50696d12ec626025927f0ac66f746721626c814c0c83aa8008485a8"
)
