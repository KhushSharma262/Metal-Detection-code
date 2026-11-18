import yaml
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _to_namespace(d):
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(ns, key, _to_namespace(value) if isinstance(value, dict) else value)
    return ns


def _resolve(p):
    p = Path(p)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def load_config(config_path=None):
    config_path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ns = _to_namespace(raw)
    ns.dataset.root_dir = _resolve(ns.dataset.root_dir)
    ns.paths.models_dir = _resolve(ns.paths.models_dir)
    ns.paths.logs_dir = _resolve(ns.paths.logs_dir)
    models_dir = Path(ns.paths.models_dir)
    ns.paths.best_model = str(models_dir / ns.paths.best_model_name)
    ns.paths.last_checkpoint = str(models_dir / ns.paths.last_checkpoint_name)
    ns.paths.class_labels = str(models_dir / ns.paths.class_labels_name)
    return ns


cfg = load_config()