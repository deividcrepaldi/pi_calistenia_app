"""
Fachada do módulo de inferência — arquitetura em CASCATA.

Dois modelos LSTM por sessão:

  detector.h5               global, compartilhado entre exercícios.
                            Decide se a janela de 30 frames contém um
                            exercício reconhecido (saída ≤ 0.5) ou algo
                            fora da distribuição de treino — pessoa
                            em pé, sentada, andando (saída > 0.5).

  <exercicio>.h5            avaliador binário correto / incorreto,
                            treinado só com amostras do exercício
                            específico.

Fluxo por janela (a cada 15 frames, depois que o buffer tiver 30):

          buffer (1, 30, 108)
                 │
                 ▼
            detector.h5
          ╱            ╲
  > 0.5                 ≤ 0.5
 "outro"             exercício
 fora_posicao             │
 (avaliador                ▼
  NÃO roda)         <exercicio>.h5
                       ╱    ╲
                   > 0.5     ≤ 0.5
                 incorreto   correto
                     │
                     ▼
                 detector de erro
                 postural por ângulo

Valores BRUTOS (coords_99 + angulos_9) vão pros dois modelos, idêntico ao
pipeline de treinamento. Ângulos SUAVIZADOS (EMA) são usados apenas para
exibição e para a regra biomecânica de erro postural.

Uso típico:
    inf = Inferencia("plank")
    resultado = inf.processar_frame(coords)  # coords pode ser None
"""

import numpy as np

from config.caminhos import caminho_detector, caminho_modelo
from config.exercicios import Exercicio, obter_exercicio
from nucleo.angulos import angulos_para_dict, calcular_9_angulos
from nucleo.buffer import BufferFrames
from nucleo.detector_erros import identificar_erro
from nucleo.modelo import ClassificadorLSTM
from nucleo.suavizacao import SuavizadorEMA


# ─── Umbral do detector global ────────────────────────────────────────
#
# > UMBRAL_DETECTOR → a janela é "outro".
#
# Por que 0.75 e não 0.5 (default sigmoide):
#   Em testes empíricos, o detector cospe valores em três faixas distintas:
#     • < 0.3   → claramente um exercício treinado (plank, push-up, etc.)
#     • 0.3-0.7 → POSE AMBÍGUA: variantes do exercício (push-up no joelho,
#                 push-up com forma ruim, plank com quadril alto), ou
#                 instantes de transição. Tecnicamente ainda é o exercício.
#     • > 0.85  → claramente NÃO é o exercício (pessoa em pé, sentada, andando).
#
#   Com umbral 0.5, qualquer pose ambígua (sigmoid ~0.6) cai no lado "outro" e
#   o banner mostra "POSIÇÃO NÃO RECONHECIDA" mesmo o usuário estando na
#   posição do exercício, só com forma incorreta — o que quebra a expectativa
#   do app (que deveria dizer INCORRETO nesses casos, não esconder o feedback).
#
#   Movendo pra 0.75, todas essas poses ambíguas viram exercício e o avaliador
#   roda normalmente. Só vira fora_posicao quando o detector está realmente
#   confiante (>0.75) — situações como o usuário sentando, levantando, saindo
#   do quadro.
#
# Histerese assimétrica (continua):
#   • exercício → outro: exige VOTOS_DETECTOR_PARA_OUTRO inferências
#     consecutivas acima do umbral. Filtra blips transitórios em exercícios
#     dinâmicos.
#   • outro → exercício: imediata.
#
# Com inferência a cada 15 frames (~0.5 s), 4 votos ≈ 2 s de "outro" alto
# consistente antes de trocar.

UMBRAL_DETECTOR = 0.75
VOTOS_DETECTOR_PARA_OUTRO = 4

# ─── Histerese / margem de confiabilidade do AVALIADOR ───────────────
#
# Para TROCAR entre correto/incorreto depois da primeira inferência,
# o sinal precisa ficar fora da "zona cinza" [UMBRAL_CORRETO, UMBRAL_INCORRETO)
# por pelo menos VOTOS_PARA_TROCAR inferências consecutivas. Na zona cinza
# a classificação e a confiança exibidas permanecem as da última inferência
# confiante.
#
# Com janela de 15 frames (~0.5 s a 30 fps), 2 votos ≈ 1 s de sinal forte
# contínuo antes de trocar — suficiente pra ignorar picos transitórios.

UMBRAL_INCORRETO   = 0.60
UMBRAL_CORRETO     = 0.40
VOTOS_PARA_TROCAR  = 2

# A suavização dos ângulos (EMA, somente exibição e detecção de erro)
# é configurada por exercício em `config.exercicios.Exercicio` —
# exercícios dinâmicos como push-up usam α maior pra acompanhar o
# movimento; exercícios estáticos usam α menor pra mais estabilidade.


class Inferencia:
    def __init__(self, exercicio_id: str):
        self._exercicio: Exercicio = obter_exercicio(exercicio_id)
        self._detector = ClassificadorLSTM(caminho_detector())
        self._avaliador = ClassificadorLSTM(
            caminho_modelo(self._exercicio.arquivo_modelo)
        )
        self._buffer = BufferFrames(tamanho=30, passo=15)
        self._suavizador_angulos = SuavizadorEMA(
            self._exercicio.alpha_suavizacao_angulos
        )

        # Estado do avaliador (correto vs incorreto).
        self._classificacao: str | None = None
        self._confianca: float = 0.0
        self._erro_nome: str | None = None
        self._erro_correcao: str | None = None
        self._erro_desvio: float | None = None
        self._votos_troca: int = 0

        # Estado do detector (em posição vs fora).
        self._fora_posicao = False
        self._confianca_detector = 0.0
        self._votos_outro = 0  # histerese exercício → outro

        self._pronto = False

    # ──────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────

    @property
    def exercicio(self) -> Exercicio:
        return self._exercicio

    def processar_frame(self, coords_99):
        """
        coords_99: ndarray (99,) de keypoints MediaPipe já no formato
                   [x, y, vis, x, y, vis, ...], ou None se o frame foi
                   descartado (sem pose ou visibilidade baixa).
        """
        angulos_dict_exibicao = None

        if coords_99 is not None:
            # Brutos: vão pros modelos (cópia do pipeline de treino).
            angulos_9_bruto = calcular_9_angulos(coords_99)

            # Suaves: vão pro painel e pra detecção de erro.
            angulos_9_suave = self._suavizador_angulos.atualizar(angulos_9_bruto)
            angulos_dict_exibicao = angulos_para_dict(angulos_9_suave)

            frame_completo = np.concatenate(
                [coords_99, angulos_9_bruto]
            ).astype(np.float32)
            self._buffer.adicionar(frame_completo)

            if self._buffer.deve_inferir():
                entrada = self._buffer.como_entrada_modelo()
                self._buffer.marcar_inferencia()

                if not self._exercicio.usa_detector:
                    # Modo "só avaliador": pula o detector. Útil pra exercícios
                    # onde o detector tem viés contra a vista (caso do side
                    # plank frontal). Confiança_detector fica zerada — nada
                    # foi medido.
                    self._confianca_detector = 0.0
                    self._votos_outro = 0
                    self._fora_posicao = False
                    saida_avaliador = self._avaliador.prever(entrada)
                    self._aplicar_avaliador(saida_avaliador, angulos_dict_exibicao)
                else:
                    # Etapa 1 — detector global.
                    saida_detector = self._detector.prever(entrada)
                    self._confianca_detector = float(saida_detector)

                    if saida_detector > UMBRAL_DETECTOR:
                        # Detector votou "outro".
                        if self._fora_posicao or self._classificacao is None:
                            # Já em fora_posicao, ou ainda não temos classe
                            # estabelecida: aplica imediatamente.
                            self._aplicar_fora_posicao()
                            self._votos_outro = 0
                        else:
                            # Em exercício com classe estabelecida: acumula voto.
                            # Só troca pra fora_posicao após VOTOS_DETECTOR_PARA_OUTRO
                            # inferências consecutivas — evita piscadas durante reps.
                            self._votos_outro += 1
                            if self._votos_outro >= VOTOS_DETECTOR_PARA_OUTRO:
                                self._aplicar_fora_posicao()
                                self._votos_outro = 0
                            # senão: mantém a classificação atual (correto/incorreto).
                            # Não consulta o avaliador — a janela é ambígua, melhor
                            # preservar o último veredito estável.
                    else:
                        # Detector votou "exercício" — zera contador e roda avaliador.
                        self._votos_outro = 0
                        saida_avaliador = self._avaliador.prever(entrada)
                        self._aplicar_avaliador(saida_avaliador, angulos_dict_exibicao)

        return {
            "pronto": self._pronto,
            "classificacao": self._classificacao,
            "confianca": self._confianca,
            "erro": self._erro_nome,
            "correcao": self._erro_correcao,
            "desvio": self._erro_desvio,
            "angulos": angulos_dict_exibicao,
            "frames_buffer": self._buffer.tamanho_atual(),
            "capacidade_buffer": self._buffer.capacidade,
            "fora_posicao": self._fora_posicao,
            "confianca_detector": self._confianca_detector,
        }

    def resetar(self):
        self._buffer.resetar()
        self._suavizador_angulos.reset()
        self._classificacao = None
        self._confianca = 0.0
        self._erro_nome = None
        self._erro_correcao = None
        self._erro_desvio = None
        self._votos_troca = 0
        self._fora_posicao = False
        self._confianca_detector = 0.0
        self._votos_outro = 0
        self._pronto = False

    # ──────────────────────────────────────────────────────────────────
    # Ramo "fora de posição" (detector decide)
    # ──────────────────────────────────────────────────────────────────

    def _aplicar_fora_posicao(self):
        """O detector classificou a janela como 'outro'. Não consulta o
        avaliador e limpa qualquer estado de correto/incorreto, para que
        na volta à postura a próxima classificação valha imediatamente
        em vez de ficar presa na votação da transição anterior."""
        self._fora_posicao = True
        self._pronto = True
        self._classificacao = None
        self._confianca = 0.0
        self._erro_nome = None
        self._erro_correcao = None
        self._erro_desvio = None
        self._votos_troca = 0

    # ──────────────────────────────────────────────────────────────────
    # Ramo "exercício detectado" — avaliador com histerese + votação
    # ──────────────────────────────────────────────────────────────────

    def _aplicar_avaliador(self, saida_sigmoide: float, angulos_dict: dict):
        self._fora_posicao = False

        voto = self._voto_da_inferencia(saida_sigmoide)

        if self._classificacao is None:
            # Primeira inferência (ou primeira depois de sair de fora_posicao):
            # compromete com o limiar simples 0.5 pra dar feedback imediato.
            # Trocas seguintes exigem histerese + votação.
            classe_inicial = "incorreto" if saida_sigmoide >= 0.5 else "correto"
            self._classificacao = classe_inicial
            self._confianca = self._confianca_para_classe(saida_sigmoide, classe_inicial)
            self._votos_troca = 0

        elif voto is None:
            # Zona cinza: mantém a classificação atual, não atualiza confiança
            # (evita flutuação visível de um número incerto).
            self._votos_troca = 0

        elif voto == self._classificacao:
            # Reconfirma a classe atual: atualiza a confiança exibida.
            self._confianca = self._confianca_para_classe(saida_sigmoide, voto)
            self._votos_troca = 0

        else:
            # Sinal forte na classe OPOSTA: acumula voto; só troca quando
            # atingir VOTOS_PARA_TROCAR inferências consecutivas.
            self._votos_troca += 1
            if self._votos_troca >= VOTOS_PARA_TROCAR:
                self._classificacao = voto
                self._confianca = self._confianca_para_classe(saida_sigmoide, voto)
                self._votos_troca = 0

        self._pronto = True

        # Erro postural acompanha a classe CONFIRMADA (sempre consistente
        # com o que está sendo exibido no card de classificação).
        if self._classificacao == "incorreto":
            (
                self._erro_nome,
                self._erro_correcao,
                self._erro_desvio,
            ) = identificar_erro(self._exercicio, angulos_dict)
        else:
            self._erro_nome = None
            self._erro_correcao = None
            self._erro_desvio = None

    @staticmethod
    def _voto_da_inferencia(saida_sigmoide: float):
        if saida_sigmoide >= UMBRAL_INCORRETO:
            return "incorreto"
        if saida_sigmoide < UMBRAL_CORRETO:
            return "correto"
        return None

    @staticmethod
    def _confianca_para_classe(saida_sigmoide: float, classe: str) -> float:
        if classe == "incorreto":
            return float(saida_sigmoide)
        return float(1.0 - saida_sigmoide)
