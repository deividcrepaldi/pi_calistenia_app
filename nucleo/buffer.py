"""
Buffer circular de frames para a janela deslizante do LSTM.
Mantém os últimos N frames (108,) e indica quando uma nova inferência
deve ser executada (a cada `passo` frames novos, após o buffer ter enchido).
"""

from collections import deque
import numpy as np


class BufferFrames:
    def __init__(self, tamanho: int = 30, passo: int = 15):
        if passo > tamanho:
            raise ValueError("passo não pode ser maior que tamanho")
        self._tamanho = tamanho
        self._passo = passo
        self._frames: deque = deque(maxlen=tamanho)
        self._frames_novos_desde_inferencia = 0

    def adicionar(self, frame_108):
        self._frames.append(np.asarray(frame_108, dtype=np.float32))
        self._frames_novos_desde_inferencia += 1

    def cheio(self) -> bool:
        return len(self._frames) == self._tamanho

    def deve_inferir(self) -> bool:
        return self.cheio() and self._frames_novos_desde_inferencia >= self._passo

    def marcar_inferencia(self):
        self._frames_novos_desde_inferencia = 0

    def como_entrada_modelo(self) -> np.ndarray:
        """Monta o tensor (1, tamanho, 108) para entrada do modelo."""
        if not self.cheio():
            raise RuntimeError("buffer ainda não está cheio")
        sequencia = np.stack(list(self._frames), axis=0)
        return np.expand_dims(sequencia, axis=0)

    def tamanho_atual(self) -> int:
        return len(self._frames)

    @property
    def capacidade(self) -> int:
        return self._tamanho

    def resetar(self):
        self._frames.clear()
        self._frames_novos_desde_inferencia = 0
