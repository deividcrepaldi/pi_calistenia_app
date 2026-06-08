"""
Registro de exercícios disponíveis.

Cada exercício declara:
  - metadados visuais (nome, tipo, vista, instrução de câmera)
  - arquivo do modelo .h5
  - quais ângulos mostrar no painel
  - regras biomecânicas de detecção de erros

Adicionar um novo exercício = adicionar uma nova instância de `Exercicio`
e registrá-la em `EXERCICIOS`. Nenhum outro módulo precisa ser alterado.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


# ─── Estruturas ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegraErro:
    """
    Uma regra biomecânica que detecta um erro postural.

    `verificar(angulos_dict)` deve retornar:
      - um float > 0  representando o desvio do limiar quando o erro está ativo
      - None          quando o erro não se aplica
    """
    nome: str
    correcao: str
    verificar: Callable[[dict], Optional[float]]


@dataclass(frozen=True)
class Exercicio:
    id: str
    nome_exibicao: str
    tipo: str                         # "dinâmico" ou "estático"
    vista: str                        # "lateral" ou "frontal"
    instrucao_camera: str
    arquivo_modelo: str               # nome do .h5 em modelos/
    angulos_exibidos: Tuple[str, ...]
    regras_erro: Tuple[RegraErro, ...]
    # Suavização EMA — α maior = mais responsivo (acompanha movimento rápido),
    # α menor = mais estável (bom pra exercícios estáticos). Aplicado SOMENTE
    # à exibição: a entrada do modelo continua bruta, idêntica ao treino.
    alpha_suavizacao_landmarks: float = 0.30
    alpha_suavizacao_angulos: float = 0.25
    # Gate do detector global ANTES de rodar o avaliador.
    # Em alguns exercícios o detector tem viés forte contra a vista (ex.: side
    # plank frontal vira "outro" com 99% de confiança mesmo o usuário estando
    # na posição). Nesses casos, ajustar umbral não resolve — o cleaner é
    # pular o detector e confiar só no avaliador. Trade-off: perde-se a
    # proteção contra entradas out-of-distribution pra esse exercício.
    usa_detector: bool = True


# ─── PUSH-UP ───────────────────────────────────────────────────────────────

def _pushup_quadril_caindo(a):
    return 160.0 - a["alinhamento"] if a["alinhamento"] < 160.0 else None

def _pushup_quadril_empinando(a):
    return a["alinhamento"] - 190.0 if a["alinhamento"] > 190.0 else None

def _pushup_cotovelo_aberto(a):
    if a["cotovelo_esq"] > 160.0 and a["cotovelo_dir"] > 160.0:
        return min(a["cotovelo_esq"], a["cotovelo_dir"]) - 160.0
    return None


PUSHUP = Exercicio(
    id="push-up",
    nome_exibicao="Push-up",
    tipo="dinâmico",
    vista="lateral",
    instrucao_camera=(
        "Posicione a câmera lateralmente, a cerca de 1,5 m do corpo, "
        "enquadrando ombros, quadril e pés."
    ),
    arquivo_modelo="push-up.h5",
    angulos_exibidos=("alinhamento", "cotovelo_esq", "cotovelo_dir"),
    regras_erro=(
        RegraErro(
            nome="Quadril caindo",
            correcao="Eleve o quadril para alinhar com os ombros",
            verificar=_pushup_quadril_caindo,
        ),
        RegraErro(
            nome="Quadril empinando",
            correcao="Abaixe o quadril para alinhar com os ombros",
            verificar=_pushup_quadril_empinando,
        ),
        RegraErro(
            nome="Cotovelo muito aberto",
            correcao="Mantenha os cotovelos próximos ao corpo",
            verificar=_pushup_cotovelo_aberto,
        ),
    ),
    # Push-up é o único exercício dinâmico — o esqueleto precisa
    # acompanhar o movimento de subida/descida. Com α=0.30 ele atrasa
    # visivelmente em repetições rápidas; α=0.55 deixa o desenho colado
    # no corpo e ainda absorve o jitter do MediaPipe. O ângulo também
    # responde mais rápido pra o cotovelo conseguir cruzar o limiar de
    # 160° no topo da repetição (caso contrário a regra "cotovelo aberto"
    # nunca dispararia em execuções aceleradas).
    alpha_suavizacao_landmarks=0.55,
    alpha_suavizacao_angulos=0.50,
)


# ─── PLANK ─────────────────────────────────────────────────────────────────

def _plank_quadril_caindo(a):
    return 160.0 - a["alinhamento"] if a["alinhamento"] < 160.0 else None

def _plank_quadril_empinando(a):
    return a["alinhamento"] - 190.0 if a["alinhamento"] > 190.0 else None

def _plank_tronco_rodado(a):
    diff = abs(a["quadril_esq"] - a["quadril_dir"])
    return diff - 15.0 if diff > 15.0 else None

def _plank_cabeca_baixa(a):
    menor = min(a["ombro_esq"], a["ombro_dir"])
    return 50.0 - menor if menor < 50.0 else None


PLANK = Exercicio(
    id="plank",
    nome_exibicao="Plank",
    tipo="estático",
    vista="lateral",
    instrucao_camera=(
        "Posicione a câmera lateralmente, a cerca de 1,5 m do corpo, "
        "enquadrando cabeça, ombros, quadril e pés."
    ),
    arquivo_modelo="plank.h5",
    angulos_exibidos=("alinhamento", "quadril_esq", "quadril_dir", "ombro_esq"),
    regras_erro=(
        RegraErro(
            nome="Quadril caindo",
            correcao="Contraia o core e eleve o quadril",
            verificar=_plank_quadril_caindo,
        ),
        RegraErro(
            nome="Quadril empinando",
            correcao="Abaixe o quadril até alinhar com o tronco",
            verificar=_plank_quadril_empinando,
        ),
        RegraErro(
            nome="Tronco rodado",
            correcao="Mantenha os quadris paralelos ao chão",
            verificar=_plank_tronco_rodado,
        ),
        RegraErro(
            nome="Cabeça baixa",
            correcao="Mantenha a cabeça alinhada com a coluna",
            verificar=_plank_cabeca_baixa,
        ),
    ),
)


# ─── SIDEPLANK ─────────────────────────────────────────────────────────────

def _sideplank_quadril_caindo(a):
    if a["quadril_esq"] < 150.0 and a["quadril_dir"] < 150.0:
        return 150.0 - min(a["quadril_esq"], a["quadril_dir"])
    return None

def _sideplank_desalinhamento(a):
    diff = abs(a["ombro_esq"] - a["ombro_dir"])
    return diff - 20.0 if diff > 20.0 else None


SIDEPLANK = Exercicio(
    id="sideplank",
    nome_exibicao="Side Plank",
    tipo="estático",
    vista="frontal",
    instrucao_camera=(
        "Posicione a câmera de frente para o corpo, a cerca de 1,5 m, "
        "enquadrando ombros e quadril."
    ),
    arquivo_modelo="sideplank.h5",
    angulos_exibidos=("quadril_esq", "quadril_dir", "ombro_esq", "ombro_dir"),
    regras_erro=(
        RegraErro(
            nome="Quadril caindo",
            correcao="Eleve o quadril lateralmente",
            verificar=_sideplank_quadril_caindo,
        ),
        RegraErro(
            nome="Corpo desalinhado",
            correcao="Alinhe os ombros verticalmente",
            verificar=_sideplank_desalinhamento,
        ),
    ),
    # O detector global classifica side plank com 99% de confiança como
    # "outro" — provavelmente porque é o único exercício de vista frontal
    # no dataset (os outros 3 são laterais) e ele aprendeu o viés. Como
    # ajuste de umbral não resolve quando o sigmoid está pegado em 0.99,
    # desligamos o gate aqui e deixamos o avaliador rodar sempre.
    usa_detector=False,
)

# ─── HOLLOW BODY HOLD ──────────────────────────────────────────────────────

def _hollowbody_joelho_dobrado(a):
    media = (a["joelho_esq"] + a["joelho_dir"]) / 2
    return 140.0 - media if media < 140.0 else None

def _hollowbody_pernas_altas(a):
    media = (a["quadril_esq"] + a["quadril_dir"]) / 2
    return media - 160.0 if media > 160.0 else None

def _hollowbody_cabeca_no_chao(a):
    menor = min(a["ombro_esq"], a["ombro_dir"])
    return 50.0 - menor if menor < 50.0 else None

def _hollowbody_bracos_dobrados(a):
    menor = min(a["cotovelo_esq"], a["cotovelo_dir"])
    return 150.0 - menor if menor < 150.0 else None


HOLLOWBODY = Exercicio(
    id="hollowbody",
    nome_exibicao="Hollow Body Hold",
    tipo="estático",
    vista="lateral",
    instrucao_camera=(
        "Deite-se de costas e posicione a câmera lateralmente, "
        "a cerca de 1,5 m do corpo, enquadrando cabeça, ombros, "
        "quadril e pés."
    ),
    arquivo_modelo="hollowbody.h5",
    angulos_exibidos=("joelho_esq", "joelho_dir", "quadril_esq", "cotovelo_esq"),
    regras_erro=(
        RegraErro(
            nome="Joelho dobrado",
            correcao="Estenda completamente as pernas",
            verificar=_hollowbody_joelho_dobrado,
        ),
        RegraErro(
            nome="Pernas muito elevadas",
            correcao="Abaixe as pernas mantendo a lombar no chão",
            verificar=_hollowbody_pernas_altas,
        ),
        RegraErro(
            nome="Cabeça apoiada no chão",
            correcao="Eleve a cabeça mantendo o queixo próximo ao peito",
            verificar=_hollowbody_cabeca_no_chao,
        ),
        RegraErro(
            nome="Braços dobrados",
            correcao="Estenda os braços completamente acima da cabeça",
            verificar=_hollowbody_bracos_dobrados,
        ),
    ),
)


# ─── Registro ──────────────────────────────────────────────────────────────

EXERCICIOS = {e.id: e for e in (PUSHUP, PLANK, SIDEPLANK, HOLLOWBODY)}


def obter_exercicio(id_: str) -> Exercicio:
    if id_ not in EXERCICIOS:
        raise KeyError(f"Exercício não registrado: {id_}")
    return EXERCICIOS[id_]


def listar_exercicios() -> Tuple[Exercicio, ...]:
    return tuple(EXERCICIOS.values())
