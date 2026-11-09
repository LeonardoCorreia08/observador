"""Configurações e validações simples do TrafficVision AI."""
from pathlib import Path
import yaml


def load_yaml(path: str) -> dict:
    """Carrega um YAML e exige que o conteúdo seja um mapeamento."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {file_path}")
    with file_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuração inválida: {file_path}")
    return data
