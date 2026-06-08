"""
Tela 1 — seleção do exercício a analisar.
Mostra os exercícios registrados em config.exercicios como cards clicáveis.

Duas formas de iniciar a análise:
  - "Iniciar análise" → câmera ao vivo (uso normal do produto)
  - "Importar vídeo"  → arquivo .mp4/.mov/etc. pra testar com material
                         pré-gravado (útil pra desenvolvimento e
                         depuração sem precisar se filmar de novo).
"""

import tkinter as tk
from tkinter import filedialog, messagebox

from config import tema
from config.exercicios import listar_exercicios
from interface.widgets.card_exercicio import CardExercicio


_EXTENSOES_VIDEO = (
    ("Vídeos", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
    ("Todos os arquivos", "*.*"),
)


class TelaSelecao(tk.Frame):
    def __init__(self, mestre, ao_iniciar_camera, ao_iniciar_video):
        super().__init__(mestre, bg=tema.FUNDO_ESCURO)
        self._ao_iniciar_camera = ao_iniciar_camera
        self._ao_iniciar_video = ao_iniciar_video
        self._exercicio_selecionado = None
        self._cards = []

        self._construir()

    # ──────────────────────────────────────────────────────────────────
    # Construção
    # ──────────────────────────────────────────────────────────────────

    def _construir(self):
        cabecalho = tk.Frame(self, bg=tema.FUNDO_ESCURO)
        cabecalho.pack(fill="x", padx=48, pady=(40, 20))

        tk.Label(
            cabecalho,
            text="Análise Biomecânica de Calistenia",
            bg=tema.FUNDO_ESCURO,
            fg=tema.TEXTO_PRINCIPAL,
            font=tema.FONTE_TITULO,
        ).pack(anchor="w")

        tk.Label(
            cabecalho,
            text=(
                "Selecione o exercício a ser analisado. O sistema identifica "
                "desvios posturais e sugere correções em tempo real."
            ),
            bg=tema.FUNDO_ESCURO,
            fg=tema.TEXTO_SECUNDARIO,
            font=tema.FONTE_SUBTITULO,
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        area_cards = tk.Frame(self, bg=tema.FUNDO_ESCURO)
        area_cards.pack(fill="x", padx=48, pady=20)

        exercicios = listar_exercicios()
        for i, exercicio in enumerate(exercicios):
            card = CardExercicio(area_cards, exercicio, self._selecionar)
            card.grid(row=0, column=i, sticky="nsew", padx=10, ipadx=8, ipady=4)
            area_cards.grid_columnconfigure(i, weight=1, uniform="card")
            self._cards.append(card)

        self._instrucao_var = tk.StringVar(
            value="Selecione um exercício para ver a orientação de posicionamento."
        )
        tk.Label(
            self,
            textvariable=self._instrucao_var,
            bg=tema.FUNDO_ESCURO,
            fg=tema.TEXTO_SECUNDARIO,
            font=tema.FONTE_SUBTITULO,
            wraplength=900,
            justify="left",
        ).pack(anchor="w", padx=48, pady=(4, 20))

        rodape = tk.Frame(self, bg=tema.FUNDO_ESCURO)
        rodape.pack(fill="x", padx=48, pady=(10, 30))

        self._botao_video = tk.Button(
            rodape,
            text="Importar vídeo",
            command=self._video_clicado,
            bg=tema.FUNDO_MEDIO,
            fg=tema.TEXTO_SUAVE,
            activebackground=tema.FUNDO_CLARO,
            activeforeground=tema.TEXTO_PRINCIPAL,
            font=tema.FONTE_BOTAO,
            relief="flat",
            bd=0,
            padx=22,
            pady=10,
            cursor="arrow",
            state="disabled",
        )
        self._botao_video.pack(side="right", padx=(8, 0))

        self._botao_iniciar = tk.Button(
            rodape,
            text="Iniciar com câmera",
            command=self._iniciar_clicado,
            bg=tema.FUNDO_MEDIO,
            fg=tema.TEXTO_SUAVE,
            activebackground=tema.AZUL_DESTAQUE,
            activeforeground=tema.TEXTO_PRINCIPAL,
            font=tema.FONTE_BOTAO,
            relief="flat",
            bd=0,
            padx=22,
            pady=10,
            cursor="arrow",
            state="disabled",
        )
        self._botao_iniciar.pack(side="right")

    # ──────────────────────────────────────────────────────────────────
    # Eventos
    # ──────────────────────────────────────────────────────────────────

    def _selecionar(self, exercicio):
        self._exercicio_selecionado = exercicio
        for card in self._cards:
            card.definir_selecao(card.exercicio.id == exercicio.id)

        self._instrucao_var.set(exercicio.instrucao_camera)
        self._botao_iniciar.configure(
            state="normal",
            bg=tema.AZUL_DESTAQUE,
            fg=tema.TEXTO_PRINCIPAL,
            cursor="hand2",
        )
        self._botao_video.configure(
            state="normal",
            fg=tema.TEXTO_PRINCIPAL,
            cursor="hand2",
        )

    def _iniciar_clicado(self):
        if self._exercicio_selecionado is None:
            return
        self._ao_iniciar_camera(self._exercicio_selecionado)

    def _video_clicado(self):
        if self._exercicio_selecionado is None:
            return
        caminho = filedialog.askopenfilename(
            parent=self,
            title=f"Escolher vídeo para análise — {self._exercicio_selecionado.nome_exibicao}",
            filetypes=list(_EXTENSOES_VIDEO),
        )
        if not caminho:
            return
        try:
            self._ao_iniciar_video(self._exercicio_selecionado, caminho)
        except Exception as exc:  # falha de roteamento — não deveria acontecer
            messagebox.showerror("Erro ao iniciar análise", str(exc), parent=self)

    def destruir(self):
        self.destroy()
