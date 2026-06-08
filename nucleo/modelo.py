"""
Carregamento e uso dos modelos LSTM treinados (.h5).

Na arquitetura em cascata existem DOIS tipos de modelo, ambos binários
com entrada (1, 30, 108) e saída sigmoide escalar:

  detector.h5          saída > 0.5 → "outro" (não é o exercício)
                       saída ≤ 0.5 → é um exercício conhecido

  <exercicio>.h5       saída > 0.5 → execução INCORRETA
                       saída ≤ 0.5 → execução CORRETA

A classe `ClassificadorLSTM` é agnóstica quanto à semântica: ela só
devolve o escalar sigmoide. Quem chama é que aplica o umbral e traduz
para a decisão correspondente.
"""

import numpy as np


class ClassificadorLSTM:
    def __init__(self, caminho_h5: str):
        # Import tardio para evitar carregar TensorFlow no startup geral da app.
        import tensorflow as tf

        self._tf = tf
        self._modelo = tf.keras.models.load_model(caminho_h5, compile=False)
        self._caminho = caminho_h5

    def prever(self, entrada_1_30_108) -> float:
        """
        entrada_1_30_108: ndarray shape (1, 30, 108)
        retorna: probabilidade sigmoide [0, 1]. A semântica depende de
        qual modelo foi carregado (ver docstring do módulo).
        """
        entrada = np.asarray(entrada_1_30_108, dtype=np.float32)
        if entrada.shape != (1, 30, 108):
            raise ValueError(
                f"Entrada inválida: esperado (1, 30, 108), recebido {entrada.shape}"
            )
        saida = self._modelo.predict(entrada, verbose=0)
        return float(saida[0, 0])

    @property
    def caminho(self) -> str:
        return self._caminho


# Alias de compatibilidade — o nome antigo ainda funciona onde for usado.
ClassificadorExercicio = ClassificadorLSTM
