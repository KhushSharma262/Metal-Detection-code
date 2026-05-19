import yaml
from pathlib import Path
from types import SimpleNamespace

def _dict_to_namespace(d):
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    ns = _dict_to_namespace(raw)
    project_root = config_path.parent
    ns.paths.models_dir   = str(project_root / "models")
    ns.paths.logs_dir     = str(project_root / "logs")
    ns.paths.best_model   = str(project_root / "models" / "best_model.pth")
    ns.paths.final_model  = str(project_root / "models" / "final_model.pth")
    ns.paths.class_labels = str(project_root / "models" / "class_labels.json")
    ns.dataset.root_dir   = str(project_root / "dataset")
    return ns

cfg = load_config()
