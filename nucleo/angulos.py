"""
Cálculo dos 9 ângulos articulares.
IMPORTANTE: este cálculo é byte a byte idêntico ao usado no treinamento
(pi_calistenia/extrair_keypoints.py). Qualquer divergência produz predições erradas.
"""

import numpy as np


# Ordem e índice de cada ângulo dentro do frame (108,)
# O array do modelo concatena [coords(99) | angulos(9)], então os ângulos ocupam 99..107.
NOMES_ANGULOS = (
    "cotovelo_esq",   # 99
    "cotovelo_dir",   # 100
    "joelho_esq",     # 101
    "joelho_dir",     # 102
    "quadril_esq",    # 103
    "quadril_dir",    # 104
    "alinhamento",    # 105
    "ombro_esq",      # 106
    "ombro_dir",      # 107
)

INDICE_ANGULO = {nome: 99 + i for i, nome in enumerate(NOMES_ANGULOS)}


# Tripletas (A, B, C) de keypoints MediaPipe — ângulo em B formado por A-B-C.
_TRIPLETAS = (
    (15, 13, 11),  # cotovelo esq
    (16, 14, 12),  # cotovelo dir
    (23, 25, 27),  # joelho esq
    (24, 26, 28),  # joelho dir
    (11, 23, 25),  # quadril esq
    (12, 24, 26),  # quadril dir
    (11, 23, 27),  # alinhamento tronco
    (13, 11, 23),  # ombro esq
    (14, 12, 24),  # ombro dir
)


def calcular_angulo(a, b, c):
    """Ângulo em B formado por A-B-C, em graus [0, 180]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    coseno = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0)))


def calcular_9_angulos(coords_flat):
    """
    Recebe array (99,) de keypoints no formato [x0,y0,vis0, x1,y1,vis1, ...]
    e retorna array (9,) de ângulos em graus na ordem de NOMES_ANGULOS.
    """
    def kp(idx):
        return (coords_flat[idx * 3], coords_flat[idx * 3 + 1])

    return np.array(
        [calcular_angulo(kp(a), kp(b), kp(c)) for a, b, c in _TRIPLETAS],
        dtype=np.float64,
    )


def angulos_para_dict(angulos_9):
    """Converte o array (9,) num dicionário nome → valor em graus."""
    return {nome: float(angulos_9[i]) for i, nome in enumerate(NOMES_ANGULOS)}
