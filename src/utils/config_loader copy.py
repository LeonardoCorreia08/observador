from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def resolve_path(path):
    return PROJECT_ROOT / path