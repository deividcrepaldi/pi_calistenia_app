"""
Janela principal da aplicação. Gerencia a troca entre as telas 1 e 2.

A análise pode ser iniciada com a webcam (uso normal) ou com um arquivo
de vídeo previamente gravado (modo de teste). A diferença entre os dois
modos fica restrita à "fonte de frames" passada à TelaAnalise — todo o
restante do pipeline é idêntico.
"""

import tkinter as tk

from captura.camera import CameraThread
from captura.video_arquivo import VideoArquivoThread
from config import tema
from interface.tela_analise import TelaAnalise
from interface.tela_selecao import TelaSelecao


class Aplicacao(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análise Biomecânica de Calistenia")
        self.geometry("1280x780")
        self.minsize(1120, 720)
        self.configure(bg=tema.FUNDO_ESCURO)

        self._tela_atual: tk.Frame | None = None
        self.protocol("WM_DELETE_WINDOW", self._fechar_janela)

        self._mostrar_selecao()

    # ──────────────────────────────────────────────────────────────────
    # Navegação
    # ──────────────────────────────────────────────────────────────────

    def _trocar_tela(self, fabrica_nova_tela):
        if self._tela_atual is not None:
            if hasattr(self._tela_atual, "destruir"):
                self._tela_atual.destruir()
            else:
                self._tela_atual.destroy()
            self._tela_atual = None

        self._tela_atual = fabrica_nova_tela()
        self._tela_atual.pack(fill="both", expand=True)

    def _mostrar_selecao(self):
        self._trocar_tela(
            lambda: TelaSelecao(
                self,
                ao_iniciar_camera=self._mostrar_analise_camera,
                ao_iniciar_video=self._mostrar_analise_video,
            )
        )

    def _mostrar_analise_camera(self, exercicio):
        self._mostrar_analise(
            exercicio,
            fabrica_fonte=lambda: CameraThread(),
            rotulo_fonte="Câmera ao vivo",
        )

    def _mostrar_analise_video(self, exercicio, caminho_video):
        self._mostrar_analise(
            exercicio,
            fabrica_fonte=lambda: VideoArquivoThread(caminho_video),
            rotulo_fonte="Vídeo importado",
        )

    def _mostrar_analise(self, exercicio, fabrica_fonte, rotulo_fonte):
        self._trocar_tela(
            lambda: TelaAnalise(
                self,
                exercicio,
                fabrica_fonte=fabrica_fonte,
                rotulo_fonte=rotulo_fonte,
                ao_voltar=self._mostrar_selecao,
                ao_encerrar=self._fechar_janela,
            )
        )

    # ──────────────────────────────────────────────────────────────────
    # Encerramento
    # ──────────────────────────────────────────────────────────────────

    def _fechar_janela(self):
        if self._tela_atual is not None and hasattr(self._tela_atual, "destruir"):
            self._tela_atual.destruir()
            self._tela_atual = None
        self.destroy()
