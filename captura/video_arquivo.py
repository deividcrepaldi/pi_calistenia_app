"""
Leitura de um arquivo de vídeo em thread separada.

Mesma interface da `CameraThread` (`iniciar`, `ler_frame`, `parar`,
`erro_abertura`) — quem usa não precisa saber se o frame veio da webcam
ou de um arquivo. Isso permite testar a aplicação com vídeos pré-gravados
sem o aluno precisar se filmar a cada iteração.

Diferenças em relação à webcam:
  - Respeita o FPS nativo do vídeo (a webcam não tem cadência fixa). Sem
    isso o vídeo "passa" muito rápido e o usuário não acompanha o feedback.
  - Ao chegar no final, REINICIA do começo (loop). É mais útil pra testes
    do que parar abruptamente.
  - O frame mais recente é mantido sob lock; a UI só vê um frame quando
    o tempo de exibição dele chegou — assim o pipeline LSTM vê o vídeo
    na cadência em que foi gravado, igual ao treinamento.
"""

import threading
import time

import cv2


class VideoArquivoThread:
    def __init__(self, caminho: str):
        self._caminho = caminho

        self._cap = None
        self._thread = None
        self._parar = threading.Event()
        self._lock = threading.Lock()
        self._ultimo_frame = None
        self._erro_abertura = None
        self._fps = 30.0
        # Sinaliza que o vídeo recomeçou desde a última leitura. Quem
        # consumir o sinal (via `consumir_recomeco`) deve descartar o
        # estado acumulado de inferência — buffer de janelas + suavizadores
        # ficam "presos" no fim da passada anterior se não resetar, e o
        # feedback demora ~2-3 s pra atualizar depois do loop.
        self._recomecou = False

    def iniciar(self) -> bool:
        cap = cv2.VideoCapture(self._caminho)
        if not cap.isOpened():
            self._erro_abertura = (
                f"Não foi possível abrir o vídeo:\n{self._caminho}"
            )
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1.0:
            self._fps = float(fps)

        self._cap = cap
        self._parar.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        intervalo = 1.0 / self._fps
        proximo_disparo = time.monotonic()

        while not self._parar.is_set():
            if self._cap is None:
                break

            ret, frame = self._cap.read()
            if not ret:
                # Fim do vídeo: rebobina e sinaliza recomeço.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with self._lock:
                    self._recomecou = True
                proximo_disparo = time.monotonic()
                continue

            with self._lock:
                self._ultimo_frame = frame

            # Cadência fixa: dorme até o próximo "tick" do FPS nativo.
            proximo_disparo += intervalo
            atraso = proximo_disparo - time.monotonic()
            if atraso > 0:
                if self._parar.wait(timeout=atraso):
                    break
            else:
                # Ficamos atrasados (decoder lento): zera o relógio e
                # segue, evitando acumular dívida temporal.
                proximo_disparo = time.monotonic()

    def ler_frame(self):
        with self._lock:
            return None if self._ultimo_frame is None else self._ultimo_frame.copy()

    def consumir_recomeco(self) -> bool:
        """Retorna True se o vídeo recomeçou desde a última chamada (e zera
        o flag). Usado pela UI pra resetar o estado de inferência no loop."""
        with self._lock:
            if self._recomecou:
                self._recomecou = False
                return True
            return False

    def parar(self):
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._ultimo_frame = None

    @property
    def erro_abertura(self):
        return self._erro_abertura
