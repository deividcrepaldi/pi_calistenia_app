"""Card que exibe a classificação atual (CORRETO/INCORRETO) com a confiança."""

import tkinter as tk

from config import tema


class CardClassificacao(tk.Frame):
    def __init__(self, mestre):
        super().__init__(
            mestre,
            bg=tema.FUNDO_CLARO,
            highlightthickness=2,
            highlightbackground=tema.BORDA,
        )
        self._texto_var = tk.StringVar(value="—")
        self._confianca_var = tk.StringVar(value="aguardando inferência")

        self._lbl_status = tk.Label(
            self,
            textvariable=self._texto_var,
            bg=tema.FUNDO_CLARO,
            fg=tema.TEXTO_SECUNDARIO,
            font=tema.FONTE_CLASSIFIC,
        )
        self._lbl_status.pack(anchor="w", padx=18, pady=(14, 2))

        self._lbl_conf = tk.Label(
            self,
            textvariable=self._confianca_var,
            bg=tema.FUNDO_CLARO,
            fg=tema.TEXTO_SECUNDARIO,
            font=tema.FONTE_CONFIANCA,
        )
        self._lbl_conf.pack(anchor="w", padx=18, pady=(0, 14))

    def mostrar_aguardando(self, frames_atual: int, capacidade: int):
        self._texto_var.set("—")
        self._confianca_var.set(f"Aguardando frames {frames_atual}/{capacidade}")
        self._aplicar_cores(tema.FUNDO_CLARO, tema.TEXTO_SECUNDARIO, tema.BORDA)

    def mostrar_fora_posicao(self, confianca_detector: float):
        self._texto_var.set("POSIÇÃO NÃO RECONHECIDA")
        self._confianca_var.set(
            f"Assuma a postura do exercício  ·  {confianca_detector * 100:.1f}%"
        )
        self._aplicar_cores(tema.FUNDO_CLARO, tema.TEXTO_SECUNDARIO, tema.BORDA)

    def mostrar_correto(self, confianca: float):
        self._texto_var.set("CORRETO")
        self._confianca_var.set(f"Confiança: {confianca * 100:.1f}%")
        self._aplicar_cores(tema.AZUL_ESCURO, tema.AZUL_CLARO, tema.AZUL_DESTAQUE)

    def mostrar_incorreto(self, confianca: float):
        self._texto_var.set("INCORRETO")
        self._confianca_var.set(f"Confiança: {confianca * 100:.1f}%")
        self._aplicar_cores(tema.VERMELHO_ESCURO, tema.VERMELHO_CLARO, tema.VERMELHO_CLARO)

    def _aplicar_cores(self, bg, fg, borda):
        self.configure(bg=bg, highlightbackground=borda)
        self._lbl_status.configure(bg=bg, fg=fg)
        self._lbl_conf.configure(bg=bg, fg=fg)
