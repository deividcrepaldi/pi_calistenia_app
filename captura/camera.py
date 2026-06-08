"""
Captura de webcam em thread separada.

A thread lê frames o mais rápido possível e mantém apenas o mais recente
disponível, evitando acumular atraso na interface. A UI consulta via
`ler_frame()` de forma não bloqueante.
"""

import threading
import cv2


class CameraThread:
    def __init__(self, indice: int = 0, largura: int = 1280, altura: int = 720):
        self._indice = indice
        self._largura = largura
        self._altura = altura

        self._cap = None
        self._thread = None
        self._parar = threading.Event()
        self._lock = threading.Lock()
        self._ultimo_frame = None
        self._erro_abertura = None

    def iniciar(self):
        # CAP_DSHOW costuma abrir a webcam mais rápido no Windows.
        cap = cv2.VideoCapture(self._indice, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._indice)
        if not cap.isOpened():
            self._erro_abertura = (
                f"Não foi possível abrir a webcam (índice {self._indice})."
            )
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._altura)
        self._cap = cap
        self._parar.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while not self._parar.is_set():
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if not ret:
                continue
            with self._lock:
                self._ultimo_frame = frame

    def ler_frame(self):
        with self._lock:
            return None if self._ultimo_frame is None else self._ultimo_frame.copy()

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
