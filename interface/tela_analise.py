"""
Tela 2 — análise em tempo real.

À esquerda, vídeo (webcam ou arquivo importado) com esqueleto MediaPipe
sobreposto e um overlay grande de status no topo. À direita, painel
com classificação e — quando aplicável — bloco de erro postural com
correção e magnitude do desvio.

A tela não decide entre webcam e vídeo importado: ela recebe uma
`fabrica_fonte` que devolve algo com a interface (`iniciar`, `ler_frame`,
`parar`, `erro_abertura`). A camada superior (Aplicacao) é quem escolhe.

Os ângulos articulares numéricos NÃO são exibidos como grade — durante
o exercício o usuário não consegue acompanhar números mudando várias
vezes por segundo. A informação angular só aparece quando é acionável:
inline com o erro detectado, na forma "desvio de N°".

Todos os valores exibidos vêm exclusivamente da inferência real do LSTM
e do cálculo real dos ângulos via MediaPipe — nada é mockado ou hardcoded.
"""

import threading
import tkinter as tk

from config import tema
from config.caminhos import caminho_pose_landmarker
from config.exercicios import Exercicio
from inferencia import Inferencia
from interface.widgets.canvas_video import CanvasVideo
from interface.widgets.card_classificacao import CardClassificacao
from nucleo.desenho import desenhar_esqueleto, desenhar_status_overlay
from nucleo.extrator import ExtratorKeypoints
from nucleo.suavizacao import SuavizadorLandmarks


INTERVALO_LOOP_MS = 15  # ~66 Hz teórico; cadência real limitada pelo MediaPipe

# A suavização dos landmarks (apenas pro desenho do esqueleto, não afeta
# o modelo) é configurada por exercício em `config.exercicios.Exercicio`.
# Exercícios dinâmicos (push-up) usam α maior pra o esqueleto acompanhar
# o movimento; estáticos (plank, sideplank, hollow body) usam α menor.


class TelaAnalise(tk.Frame):
    def __init__(
        self,
        mestre,
        exercicio: Exercicio,
        fabrica_fonte,
        rotulo_fonte: str,
        ao_voltar,
        ao_encerrar,
    ):
        super().__init__(mestre, bg=tema.FUNDO_ESCURO)
        self._exercicio = exercicio
        self._fabrica_fonte = fabrica_fonte
        self._rotulo_fonte = rotulo_fonte
        self._ao_voltar = ao_voltar
        self._ao_encerrar = ao_encerrar

        self._fonte = None  # CameraThread ou VideoArquivoThread
        self._extrator: ExtratorKeypoints | None = None
        self._inferencia: Inferencia | None = None
        self._suavizador_landmarks = SuavizadorLandmarks(
            alpha=exercicio.alpha_suavizacao_landmarks
        )

        self._loop_ativo = False
        self._id_after: str | None = None
        self._encerrando = False

        self._construir_layout()
        self._iniciar_carregamento()

    # ──────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────

    def _construir_layout(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1, minsize=360)
        self.grid_rowconfigure(0, weight=1)

        # ── vídeo (esquerda) ─────────────────────────────────────────
        area_video = tk.Frame(self, bg=tema.FUNDO_ESCURO)
        area_video.grid(row=0, column=0, sticky="nsew", padx=(24, 12), pady=24)
        area_video.grid_rowconfigure(0, weight=1)
        area_video.grid_columnconfigure(0, weight=1)

        self._canvas_video = CanvasVideo(area_video)
        self._canvas_video.grid(row=0, column=0, sticky="nsew")
        self._canvas_video.exibir_mensagem("Inicializando câmera e modelos...")

        # ── painel (direita) ─────────────────────────────────────────
        painel = tk.Frame(self, bg=tema.FUNDO_ESCURO)
        painel.grid(row=0, column=1, sticky="nsew", padx=(12, 24), pady=24)
        painel.grid_columnconfigure(0, weight=1)
        # O bloco de erro fica numa linha que pode crescer; o card de
        # classificação ocupa o topo, os botões ficam embaixo.
        painel.grid_rowconfigure(2, weight=1)

        self._construir_cabecalho(painel, linha=0)
        self._construir_card_classificacao(painel, linha=1)
        self._construir_bloco_erro(painel, linha=2)
        self._construir_botoes(painel, linha=3)

    def _construir_cabecalho(self, pai, linha):
        cab = tk.Frame(pai, bg=tema.FUNDO_ESCURO)
        cab.grid(row=linha, column=0, sticky="ew", pady=(0, 14))

        tk.Label(
            cab,
            text=self._exercicio.nome_exibicao,
            bg=tema.FUNDO_ESCURO,
            fg=tema.TEXTO_PRINCIPAL,
            font=tema.FONTE_TITULO,
        ).pack(anchor="w")

        meta = (
            f"{self._exercicio.tipo.capitalize()}  ·  vista {self._exercicio.vista}"
            f"  ·  {self._rotulo_fonte}"
        )
        tk.Label(
            cab,
            text=meta,
            bg=tema.FUNDO_ESCURO,
            fg=tema.TEXTO_SECUNDARIO,
            font=tema.FONTE_CARD_META,
        ).pack(anchor="w")

    def _construir_card_classificacao(self, pai, linha):
        self._card_class = CardClassificacao(pai)
        self._card_class.grid(row=linha, column=0, sticky="ew", pady=(0, 14))
        self._card_class.mostrar_aguardando(0, 30)

    def _construir_bloco_erro(self, pai, linha):
        self._bloco_erro = tk.Frame(
            pai,
            bg=tema.FUNDO_MEDIO,
            highlightthickness=2,
            highlightbackground=tema.VERMELHO_CLARO,
        )
        # não dá grid ainda; só aparece quando há erro

        self._erro_nome_var = tk.StringVar(value="")
        self._erro_correcao_var = tk.StringVar(value="")
        self._erro_desvio_var = tk.StringVar(value="")

        tk.Label(
            self._bloco_erro,
            textvariable=self._erro_nome_var,
            bg=tema.FUNDO_MEDIO,
            fg=tema.VERMELHO_CLARO,
            font=tema.FONTE_ERRO,
            anchor="w",
            justify="left",
            wraplength=320,
        ).pack(anchor="w", padx=16, pady=(14, 4), fill="x")

        tk.Label(
            self._bloco_erro,
            textvariable=self._erro_desvio_var,
            bg=tema.FUNDO_MEDIO,
            fg=tema.TEXTO_SECUNDARIO,
            font=tema.FONTE_CARD_META,
            anchor="w",
            justify="left",
            wraplength=320,
        ).pack(anchor="w", padx=16, pady=(0, 8), fill="x")

        tk.Label(
            self._bloco_erro,
            textvariable=self._erro_correcao_var,
            bg=tema.FUNDO_MEDIO,
            fg=tema.TEXTO_PRINCIPAL,
            font=tema.FONTE_CORRECAO,
            anchor="w",
            justify="left",
            wraplength=320,
        ).pack(anchor="w", padx=16, pady=(0, 14), fill="x")

        self._linha_bloco_erro = linha
        self._bloco_erro_visivel = False

    def _construir_botoes(self, pai, linha):
        rodape = tk.Frame(pai, bg=tema.FUNDO_ESCURO)
        rodape.grid(row=linha, column=0, sticky="ew", pady=(14, 0))

        tk.Button(
            rodape,
            text="Voltar",
            command=self._voltar_clicado,
            bg=tema.FUNDO_MEDIO,
            fg=tema.TEXTO_PRINCIPAL,
            activebackground=tema.FUNDO_CLARO,
            activeforeground=tema.TEXTO_PRINCIPAL,
            font=tema.FONTE_BOTAO,
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            cursor="hand2",
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        tk.Button(
            rodape,
            text="Encerrar",
            command=self._encerrar_clicado,
            bg=tema.VERMELHO_ESCURO,
            fg=tema.VERMELHO_CLARO,
            activebackground=tema.VERMELHO_CLARO,
            activeforeground=tema.FUNDO_ESCURO,
            font=tema.FONTE_BOTAO,
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            cursor="hand2",
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

    # ──────────────────────────────────────────────────────────────────
    # Ciclo de vida
    # ──────────────────────────────────────────────────────────────────

    def _iniciar_carregamento(self):
        threading.Thread(target=self._carregar, daemon=True).start()

    def _carregar(self):
        """Executa em thread para não bloquear a UI com imports pesados."""
        inferencia = None
        extrator = None
        fonte = None
        try:
            if self._encerrando:
                return
            inferencia = Inferencia(self._exercicio.id)

            if self._encerrando:
                return
            extrator = ExtratorKeypoints(caminho_pose_landmarker())

            if self._encerrando:
                extrator.fechar()
                return
            fonte = self._fabrica_fonte()
            if not fonte.iniciar():
                raise RuntimeError(
                    fonte.erro_abertura or "Falha ao iniciar a fonte de vídeo."
                )
        except Exception as exc:
            mensagem = str(exc)
            if fonte is not None:
                fonte.parar()
            if extrator is not None:
                extrator.fechar()
            self._agendar_seguro(lambda: self._exibir_erro_fatal(mensagem))
            return

        self._agendar_seguro(lambda: self._publicar(inferencia, extrator, fonte))

    def _publicar(self, inferencia, extrator, fonte):
        if self._encerrando:
            fonte.parar()
            extrator.fechar()
            return
        self._inferencia = inferencia
        self._extrator = extrator
        self._fonte = fonte
        self._loop_ativo = True
        self._loop()

    def _agendar_seguro(self, callback):
        """Agenda callback na main loop; ignora se a tela já foi destruída."""
        try:
            self.after(0, callback)
        except Exception:
            pass

    def _loop(self):
        if not self._loop_ativo or self._encerrando:
            return

        try:
            # Vídeo importado loopa ao chegar no fim. Sem reset, o buffer de
            # 30 frames carrega frames do fim da passada anterior pro início
            # da nova — feedback demora 2-3 s pra atualizar. Camera_thread
            # não tem esse método (não loopa), então nada acontece pra webcam.
            consumir = getattr(self._fonte, "consumir_recomeco", None)
            if consumir is not None and consumir():
                self._inferencia.resetar()
                self._suavizador_landmarks.reset()

            frame = self._fonte.ler_frame()
            if frame is not None:
                landmarks, coords = self._extrator.detectar(frame)
                resultado = self._inferencia.processar_frame(coords)
                landmarks_para_desenho = self._suavizador_landmarks.atualizar(landmarks)
                desenhar_esqueleto(frame, landmarks_para_desenho)
                desenhar_status_overlay(frame, resultado, self._rotulo_fonte)
                self._canvas_video.exibir_frame(frame)
                self._atualizar_painel(resultado)
        except Exception as exc:
            # log discreto no stderr para não quebrar o loop
            import sys
            print(f"[loop] erro: {exc}", file=sys.stderr)

        self._id_after = self.after(INTERVALO_LOOP_MS, self._loop)

    # ──────────────────────────────────────────────────────────────────
    # Atualização do painel
    # ──────────────────────────────────────────────────────────────────

    def _atualizar_painel(self, resultado):
        if not resultado["pronto"]:
            self._card_class.mostrar_aguardando(
                resultado["frames_buffer"], resultado["capacidade_buffer"]
            )
            self._ocultar_bloco_erro()
        elif resultado["fora_posicao"]:
            self._card_class.mostrar_fora_posicao(resultado["confianca_detector"])
            self._ocultar_bloco_erro()
        elif resultado["classificacao"] == "correto":
            self._card_class.mostrar_correto(resultado["confianca"])
            self._ocultar_bloco_erro()
        else:
            self._card_class.mostrar_incorreto(resultado["confianca"])
            if resultado["erro"]:
                self._exibir_bloco_erro(
                    resultado["erro"],
                    resultado["correcao"],
                    resultado.get("desvio"),
                )
            else:
                self._ocultar_bloco_erro()

    def _exibir_bloco_erro(self, nome_erro, correcao, desvio):
        self._erro_nome_var.set(f"⚠ {nome_erro}")
        if desvio is not None:
            self._erro_desvio_var.set(f"Desvio detectado: {desvio:.1f}°")
        else:
            self._erro_desvio_var.set("")
        self._erro_correcao_var.set(correcao or "")
        if not self._bloco_erro_visivel:
            self._bloco_erro.grid(
                row=self._linha_bloco_erro, column=0, sticky="new", pady=(0, 10)
            )
            self._bloco_erro_visivel = True

    def _ocultar_bloco_erro(self):
        if self._bloco_erro_visivel:
            self._bloco_erro.grid_remove()
            self._bloco_erro_visivel = False

    def _exibir_erro_fatal(self, mensagem):
        self._canvas_video.exibir_mensagem(
            f"Não foi possível iniciar a análise:\n{mensagem}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Eventos
    # ──────────────────────────────────────────────────────────────────

    def _voltar_clicado(self):
        self._parar_recursos()
        self._ao_voltar()

    def _encerrar_clicado(self):
        self._parar_recursos()
        self._ao_encerrar()

    # ──────────────────────────────────────────────────────────────────
    # Encerramento / limpeza
    # ──────────────────────────────────────────────────────────────────

    def _parar_recursos(self):
        self._encerrando = True
        self._loop_ativo = False

        if self._id_after is not None:
            try:
                self.after_cancel(self._id_after)
            except Exception:
                pass
            self._id_after = None

        if self._fonte is not None:
            self._fonte.parar()
            self._fonte = None

        if self._extrator is not None:
            self._extrator.fechar()
            self._extrator = None

        if self._inferencia is not None:
            self._inferencia.resetar()
            self._inferencia = None

        self._suavizador_landmarks.reset()

    def destruir(self):
        self._parar_recursos()
        self.destroy()
