from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"  # Mantido o diretório "configs" do padrão original


def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Nova validação adicionada
    if not isinstance(config, dict):
        raise ValueError("Configuração YAML inválida.")

    return config


def resolve_path(path):
    # Nova lógica para suportar caminhos absolutos mantendo a compatibilidade
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


# ==========================================
# Novas funções adicionadas solicitadas
# ==========================================

def get_experiment(config, experiment_id):
    experiments = config.get("experiments", {})

    if experiment_id not in experiments:
        raise ValueError(
            f"Experimento não encontrado: {experiment_id}"
        )

    experiment = experiments[experiment_id]
    experiment["output_path"] = resolve_path(
        experiment["output_dir"]
    )

    return experiment


def get_device(config):
    device = config.get("runtime", {}).get("device", "auto")

    if device != "auto":
        return device

    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass

    return "cpu"