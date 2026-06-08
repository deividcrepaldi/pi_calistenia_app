"""
Suavização temporal (EMA) para estabilizar o que é mostrado ao usuário.

Aplicado APENAS a valores de exibição — ângulos no painel e posições dos
keypoints desenhados sobre o vídeo — e ao teste de limiar que identifica
o erro postural. O tensor que entra no modelo LSTM continua sendo montado
com coordenadas e ângulos BRUTOS, mantendo o pipeline byte a byte idêntico
ao do treinamento (qualquer divergência invalidaria as predições).

Filtro usado:
    EMA: estado_novo = α · atual + (1 − α) · estado_anterior

α menor → mais suavização (visual estável, mas lento para responder)
α maior → mais responsividade (rápido, mas mais trêmulo)
"""

import numpy as np


class SuavizadorEMA:
    """EMA vetorial aplicado a qualquer ndarray de shape fixo."""

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha deve estar em (0, 1]")
        self._alpha = alpha
        self._estado = None

    def atualizar(self, valor):
        if valor is None:
            return None if self._estado is None else self._estado.copy()
        atual = np.asarray(valor, dtype=np.float64)
        if self._estado is None or self._estado.shape != atual.shape:
            self._estado = atual.copy()
        else:
            self._estado = self._alpha * atual + (1.0 - self._alpha) * self._estado
        return self._estado.copy()

    def reset(self):
        self._estado = None


class _LandmarkSuavizado:
    """Proxy mínimo com os campos que o desenhador do esqueleto usa."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility


class SuavizadorLandmarks:
    """
    EMA sobre os 33 landmarks do MediaPipe Pose (x, y, visibility).
    A saída mantém a interface (.x, .y, .visibility) esperada por
    `nucleo.desenho.desenhar_esqueleto`.

    Se a detecção falha (landmarks=None), devolve None — o desenhador
    não desenha nada nesse frame.
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha deve estar em (0, 1]")
        self._alpha = alpha
        self._estado = None  # ndarray (33, 3)

    def atualizar(self, landmarks):
        if landmarks is None:
            return None

        atual = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in landmarks],
            dtype=np.float64,
        )

        if self._estado is None or self._estado.shape != atual.shape:
            self._estado = atual.copy()
        else:
            self._estado = self._alpha * atual + (1.0 - self._alpha) * self._estado

        return [_LandmarkSuavizado(r[0], r[1], r[2]) for r in self._estado]

    def reset(self):
        self._estado = None
