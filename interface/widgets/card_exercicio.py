"""Card clicável para seleção de exercício na tela inicial."""

import tkinter as tk

from config import tema
from config.exercicios import Exercicio


class CardExercicio(tk.Frame):
    def __init__(self, mestre, exercicio: Exercicio, ao_clicar):
        super().__init__(
            mestre,
            bg=tema.FUNDO_MEDIO,
            highlightthickness=2,
            highlightbackground=tema.BORDA,
            cursor="hand2",
        )
        self.exercicio = exercicio
        self._ao_clicar = ao_clicar
        self._selecionado = False

        self._construir()
        self._registrar_cliques(self)

    def _construir(self):
        titulo = tk.Label(
            self,
            text=self.exercicio.nome_exibicao,
            bg=tema.FUNDO_MEDIO,
            fg=tema.TEXTO_PRINCIPAL,
            font=tema.FONTE_CARD_NOME,
        )
        titulo.pack(anchor="w", padx=18, pady=(18, 6))

        meta = f"{self.exercicio.tipo.capitalize()} • vista {self.exercicio.vista}"
        sub = tk.Label(
            self,
            text=meta,
            bg=tema.FUNDO_MEDIO,
            fg=tema.TEXTO_SECUNDARIO,
            font=tema.FONTE_CARD_META,
        )
        sub.pack(anchor="w", padx=18, pady=(0, 18))

    def _registrar_cliques(self, widget):
        widget.bind("<Button-1>", self._onclick)
        for filho in widget.winfo_children():
            self._registrar_cliques(filho)

    def _onclick(self, _evento):
        self._ao_clicar(self.exercicio)

    def definir_selecao(self, selecionado: bool):
        self._selecionado = selecionado
        self.configure(
            highlightbackground=tema.AZUL_DESTAQUE if selecionado else tema.BORDA
        )
