"""
Resolução de caminhos absolutos para modelos e assets.
Usa a raiz do projeto como base para evitar depender do diretório de execução.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DIR_MODELOS = RAIZ / "modelos"
DIR_ASSETS  = RAIZ / "assets"

ARQUIVO_POSE_LANDMARKER = DIR_ASSETS / "pose_landmarker_full.task"
ARQUIVO_DETECTOR = "detector.h5"  # detector global: exercício vs "outro"


def caminho_modelo(nome_arquivo: str) -> str:
    caminho = DIR_MODELOS / nome_arquivo
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {caminho}")
    return str(caminho)


def caminho_detector() -> str:
    return caminho_modelo(ARQUIVO_DETECTOR)


def caminho_pose_landmarker() -> str:
    if not ARQUIVO_POSE_LANDMARKER.exists():
        raise FileNotFoundError(
            f"pose_landmarker_full.task não encontrado em {ARQUIVO_POSE_LANDMARKER}"
        )
    return str(ARQUIVO_POSE_LANDMARKER)
