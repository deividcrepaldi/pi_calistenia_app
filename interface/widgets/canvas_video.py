"""
Canvas que exibe um frame OpenCV (BGR) redimensionado.
Usa PIL/Pillow para converter o ndarray em PhotoImage compatível com Tkinter.
"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk

from config import tema


class CanvasVideo(tk.Label):
    def __init__(self, mestre, largura: int = 820, altura: int = 615):
        super().__init__(
            mestre,
            bg=tema.FUNDO_ESCURO,
            text="",
        )
        self._largura = largura
        self._altura = altura
        self._foto = None  # manter referência, senão o Tk coleta

    def exibir_frame(self, frame_bgr):
        if frame_bgr is None:
            return

        largura_destino = max(1, self.winfo_width() or self._largura)
        altura_destino  = max(1, self.winfo_height() or self._altura)

        h, w = frame_bgr.shape[:2]
        escala = min(largura_destino / w, altura_destino / h)
        nova_w = max(1, int(w * escala))
        nova_h = max(1, int(h * escala))

        redim = cv2.resize(frame_bgr, (nova_w, nova_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(redim, cv2.COLOR_BGR2RGB)

        imagem = Image.fromarray(rgb)
        self._foto = ImageTk.PhotoImage(imagem)
        self.configure(image=self._foto)

    def exibir_mensagem(self, texto: str):
        self._foto = None
        self.configure(
            image="",
            text=texto,
            fg=tema.TEXTO_SECUNDARIO,
            font=tema.FONTE_SUBTITULO,
        )
