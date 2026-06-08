"""
Avaliação em lote do pipeline.

Roda os vídeos da pasta `testes/videos/` pelo MESMO pipeline da aplicação
(extrator MediaPipe + cascata detector → avaliador) e gera relatório com
acurácia, precisão, recall, F1 e matrizes de confusão.

Estrutura esperada (rótulos vêm do caminho — sem JSON sidecar):

    testes/videos/
        push-up/correto/*.mp4
        push-up/incorreto/*.mp4
        plank/correto/*.mp4
        plank/incorreto/*.mp4
        sideplank/correto/*.mp4
        sideplank/incorreto/*.mp4
        hollowbody/correto/*.mp4
        hollowbody/incorreto/*.mp4
        outro/*.mp4                     # vídeos que NÃO são exercício

A pasta `outro/` mede o detector global isoladamente (deveria sempre
disparar fora_posicao). As pastas de exercício medem o avaliador quando
o detector deixa passar.

Métricas são reportadas em DOIS níveis:
  • Por janela  → cada inferência (1 s ≈ 1 janela) é uma amostra. Métrica
                  mais próxima da acurácia de treino do LSTM.
  • Por vídeo   → voto majoritário das janelas. Métrica "user-facing":
                  reflete o que um avaliador humano veria como veredito
                  do vídeo inteiro.

Convenções (positivo = classe minoritária / "alerta"):
  Detector:   positivo = "outro"           (queremos detectar anomalia)
  Avaliador:  positivo = "incorreto"       (queremos detectar erro postural)

Uso:
    python testes/avaliar.py
    python testes/avaliar.py --videos minha/pasta --saida outra/pasta
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Silencia logs do TF antes do import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Permite rodar de qualquer pasta — a raiz da app fica no sys.path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config.caminhos import (  # noqa: E402
    caminho_detector,
    caminho_modelo,
    caminho_pose_landmarker,
)
from config.exercicios import listar_exercicios, obter_exercicio  # noqa: E402
from nucleo.angulos import calcular_9_angulos  # noqa: E402
from nucleo.buffer import BufferFrames  # noqa: E402
from nucleo.extrator import ExtratorKeypoints  # noqa: E402
from nucleo.modelo import ClassificadorLSTM  # noqa: E402


EXTENSOES_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
EXERCICIOS_IDS = tuple(e.id for e in listar_exercicios())
UMBRAL = 0.5


# ─── Coleta de janelas ────────────────────────────────────────────────


def coletar_janelas(extrator: ExtratorKeypoints, video_path: Path):
    """Lê o vídeo até o fim e devolve as janelas (1, 30, 108) geradas
    exatamente como na app rodando (passo=15 entre inferências)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    buffer = BufferFrames(tamanho=30, passo=15)
    janelas = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, coords = extrator.detectar(frame)
        if coords is None:
            continue
        angulos = calcular_9_angulos(coords)
        completo = np.concatenate([coords, angulos]).astype(np.float32)
        buffer.adicionar(completo)
        if buffer.deve_inferir():
            janelas.append(buffer.como_entrada_modelo())
            buffer.marcar_inferencia()

    cap.release()
    return janelas


def descobrir_videos(base: Path):
    """Yields (caminho, exercicio_pasta, classe) — `classe` é None para 'outro'."""
    if not base.exists():
        return
    for video in sorted(base.rglob("*")):
        if video.is_dir() or video.suffix.lower() not in EXTENSOES_VIDEO:
            continue
        rel = video.relative_to(base)
        partes = rel.parts
        if len(partes) < 2:
            continue
        pasta_exercicio = partes[0]
        if pasta_exercicio == "outro":
            yield video, "outro", None
        elif (
            pasta_exercicio in EXERCICIOS_IDS
            and len(partes) >= 3
            and partes[1] in ("correto", "incorreto")
        ):
            yield video, pasta_exercicio, partes[1]
        # outros casos: ignora silenciosamente


# ─── Métricas ─────────────────────────────────────────────────────────


def _veredito_janela(reg: dict) -> str:
    """O que o sistema 'mostraria' para essa janela, end-to-end:
       'outro' se o detector bloquear, senão 'correto' ou 'incorreto'."""
    if reg["previsto_detector"] == "outro":
        return "outro"
    return reg["previsto_avaliador"]


def _rotulo_real(reg: dict) -> str:
    """O rótulo verdadeiro derivado do caminho do vídeo."""
    if not reg["verdade_eh_exercicio"]:
        return "outro"
    return "incorreto" if reg["verdade_eh_incorreto"] else "correto"


def computar_eficacia(registros):
    """Eficácia end-to-end por (exercício, classe).

    Para cada vídeo, o veredito final é o que o usuário veria no banner
    (voto majoritário entre janelas). 'Acerto' = veredito bate com o
    rótulo do vídeo. Também conta acertos no nível janela pra dar
    resolução estatística maior quando o n_videos for pequeno.

    Devolve dict no formato:
        {
          (exercicio, classe): {
              'videos': N,
              'videos_acerto': K,
              'janelas': M,
              'janelas_acerto': L,
              'videos_errados': [lista de paths]
          }
        }
    Onde `classe` é 'correto', 'incorreto' ou 'outro'.
    """
    from collections import Counter

    grupos = defaultdict(lambda: {
        "videos": 0, "videos_acerto": 0,
        "janelas": 0, "janelas_acerto": 0,
        "videos_errados": [],
    })

    # Agrupa janelas por vídeo
    por_video = defaultdict(list)
    for r in registros:
        por_video[r["video"]].append(r)

    # Eficácia por vídeo (voto majoritário das janelas)
    for video, regs in por_video.items():
        rotulo = _rotulo_real(regs[0])
        ex = regs[0]["exercicio_pasta"]
        cl = regs[0]["classe"] or "outro"
        chave = (ex, cl)

        veredito = Counter(_veredito_janela(r) for r in regs).most_common(1)[0][0]
        grupos[chave]["videos"] += 1
        if veredito == rotulo:
            grupos[chave]["videos_acerto"] += 1
        else:
            grupos[chave]["videos_errados"].append({
                "video": video,
                "esperado": rotulo,
                "veredito": veredito,
            })

    # Eficácia por janela
    for reg in registros:
        ex = reg["exercicio_pasta"]
        cl = reg["classe"] or "outro"
        chave = (ex, cl)
        grupos[chave]["janelas"] += 1
        if _veredito_janela(reg) == _rotulo_real(reg):
            grupos[chave]["janelas_acerto"] += 1

    return grupos


def metricas_binarias(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Acurácia, precisão, recall e F1 a partir da matriz 2×2."""
    total = tp + fp + tn + fn
    acuracia = (tp + tn) / total if total else 0.0
    precisao = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) else 0.0
    return {
        "amostras": total,
        "acuracia": acuracia,
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def matriz_detector_por_janela(registros):
    tp = fp = tn = fn = 0
    for r in registros:
        positivo_real = not r["verdade_eh_exercicio"]            # outro
        positivo_prev = r["previsto_detector"] == "outro"
        if positivo_real and positivo_prev:       tp += 1
        elif positivo_real and not positivo_prev: fn += 1
        elif not positivo_real and positivo_prev: fp += 1
        else:                                     tn += 1
    return metricas_binarias(tp, fp, tn, fn)


def matriz_detector_por_video(registros):
    por_video = defaultdict(list)
    for r in registros:
        por_video[r["video"]].append(r)
    tp = fp = tn = fn = 0
    for video, regs in por_video.items():
        positivo_real = not regs[0]["verdade_eh_exercicio"]
        n_outro = sum(1 for r in regs if r["previsto_detector"] == "outro")
        positivo_prev = n_outro > len(regs) / 2                 # voto majoritário
        if positivo_real and positivo_prev:       tp += 1
        elif positivo_real and not positivo_prev: fn += 1
        elif not positivo_real and positivo_prev: fp += 1
        else:                                     tn += 1
    return metricas_binarias(tp, fp, tn, fn)


def matriz_avaliador_por_janela(registros, eid: str):
    tp = fp = tn = fn = 0
    for r in registros:
        if r["exercicio_pasta"] != eid or r["previsto_avaliador"] is None:
            continue
        positivo_real = r["verdade_eh_incorreto"]
        positivo_prev = r["previsto_avaliador"] == "incorreto"
        if positivo_real and positivo_prev:       tp += 1
        elif positivo_real and not positivo_prev: fn += 1
        elif not positivo_real and positivo_prev: fp += 1
        else:                                     tn += 1
    return metricas_binarias(tp, fp, tn, fn)


def matriz_avaliador_por_video(registros, eid: str):
    por_video = defaultdict(list)
    for r in registros:
        if r["exercicio_pasta"] != eid or r["previsto_avaliador"] is None:
            continue
        por_video[r["video"]].append(r)
    tp = fp = tn = fn = 0
    for video, regs in por_video.items():
        positivo_real = regs[0]["verdade_eh_incorreto"]
        n_inc = sum(1 for r in regs if r["previsto_avaliador"] == "incorreto")
        positivo_prev = n_inc > len(regs) / 2
        if positivo_real and positivo_prev:       tp += 1
        elif positivo_real and not positivo_prev: fn += 1
        elif not positivo_real and positivo_prev: fp += 1
        else:                                     tn += 1
    return metricas_binarias(tp, fp, tn, fn)


def computar_relatorio(registros, videos_processados, videos_sem_janela):
    # Eficácia (visão simples, é o que vai no Markdown principal).
    # Tupla-chave não serializa em JSON; converto pra lista de dicts.
    eficacia = computar_eficacia(registros)
    eficacia_lista = []
    for (ex, cl), dados in eficacia.items():
        eficacia_lista.append({"exercicio": ex, "classe": cl, **dados})

    relatorio = {
        "geral": {
            "videos_processados": videos_processados,
            "videos_sem_janela": videos_sem_janela,
            "janelas_total": len(registros),
        },
        "eficacia": eficacia_lista,
        # Métricas detalhadas — ficam no JSON pra quem quiser calcular F1
        # depois, mas não entram no Markdown.
        "detalhado": {
            "detector": {
                "convencao": "positivo = 'outro'",
                "por_janela": matriz_detector_por_janela(registros),
                "por_video":  matriz_detector_por_video(registros),
            },
            "avaliadores": {
                eid: {
                    "convencao": "positivo = 'incorreto'",
                    "por_janela": matriz_avaliador_por_janela(registros, eid),
                    "por_video":  matriz_avaliador_por_video(registros, eid),
                }
                for eid in EXERCICIOS_IDS
            },
        },
    }
    return relatorio


# ─── Saída Markdown ───────────────────────────────────────────────────


_NOMES_EXIBICAO = {
    "push-up":    "Push-up",
    "plank":      "Plank",
    "sideplank":  "Side Plank",
    "hollowbody": "Hollow Body Hold",
    "outro":      "Outro (não-exercício)",
}

_ROTULOS_CLASSE = {
    "correto":   "corretos",
    "incorreto": "incorretos",
    "outro":     "não-exercício",
}


def _pct(num, den):
    return (num / den * 100.0) if den else 0.0


def _ordem(eficacia_lista):
    """Ordena as linhas: push-up, plank, sideplank, hollowbody, outro,
    e dentro de cada exercício, 'correto' antes de 'incorreto'."""
    ordem_ex = {eid: i for i, eid in enumerate(EXERCICIOS_IDS)}
    ordem_ex["outro"] = len(EXERCICIOS_IDS)
    ordem_cl = {"correto": 0, "incorreto": 1, "outro": 2}
    return sorted(
        eficacia_lista,
        key=lambda x: (ordem_ex.get(x["exercicio"], 99), ordem_cl.get(x["classe"], 99)),
    )


def gerar_markdown(relatorio: dict, timestamp: str) -> str:
    linhas = []
    add = linhas.append

    add(f"# Avaliação em lote — {timestamp}")
    add("")
    g = relatorio["geral"]
    add(f"- Vídeos processados: **{g['videos_processados']}**")
    if g["videos_sem_janela"]:
        add(f"- Vídeos descartados (pose não detectada o suficiente): {g['videos_sem_janela']}")
    add(f"- Janelas totais inferidas: **{g['janelas_total']}**")
    add("")
    add("Cada categoria (corretos/incorretos) reporta o **recall** (sensibilidade) da classe — `acertos / total da classe`. A coluna de acurácia agrega as duas categorias do exercício. A linha de Total apresenta a acurácia geral do sistema. O veredito de um vídeo é o voto majoritário entre suas janelas; o pipeline considerado é o completo (detector + avaliador).")
    add("")

    eficacia = _ordem(relatorio["eficacia"])

    # ─── Tabela única: uma linha por exercício ───────────────────────
    add("## Resultados")
    add("")
    add("| Exercício | Acerto em corretos | Acerto em incorretos | Acurácia |")
    add("|---|:-:|:-:|:-:|")

    # Agrupa por exercício (some os correto + incorreto pra calcular acurácia).
    por_ex = defaultdict(lambda: {
        "cor_v": 0, "cor_a": 0,
        "inc_v": 0, "inc_a": 0,
    })
    for g_ in eficacia:
        ex = g_["exercicio"]
        cl = g_["classe"]
        if cl == "correto":
            por_ex[ex]["cor_v"] += g_["videos"]
            por_ex[ex]["cor_a"] += g_["videos_acerto"]
        elif cl == "incorreto":
            por_ex[ex]["inc_v"] += g_["videos"]
            por_ex[ex]["inc_a"] += g_["videos_acerto"]
        elif cl == "outro":
            # Categoria única — vai no lugar de "incorretos" só pra alinhar,
            # mas o nome do exercício deixa claro.
            por_ex[ex]["inc_v"] += g_["videos"]
            por_ex[ex]["inc_a"] += g_["videos_acerto"]

    ordem_ex_full = list(EXERCICIOS_IDS) + ["outro"]
    tot_v = tot_a = 0
    for eid in ordem_ex_full:
        if eid not in por_ex:
            continue
        d = por_ex[eid]
        nome = _NOMES_EXIBICAO.get(eid, eid)
        if eid == "outro":
            # outro só tem uma categoria — junta na coluna de "incorretos" pra
            # alinhar visualmente, mas mostra como única.
            cor_txt = "—"
            inc_txt = f"{d['inc_a']}/{d['inc_v']} ({_pct(d['inc_a'], d['inc_v']):.1f}%)"
        else:
            cor_txt = f"{d['cor_a']}/{d['cor_v']} ({_pct(d['cor_a'], d['cor_v']):.1f}%)"
            inc_txt = f"{d['inc_a']}/{d['inc_v']} ({_pct(d['inc_a'], d['inc_v']):.1f}%)"

        v_ex = d["cor_v"] + d["inc_v"]
        a_ex = d["cor_a"] + d["inc_a"]
        acu_txt = f"**{_pct(a_ex, v_ex):.1f}%**"

        tot_v += v_ex
        tot_a += a_ex

        add(f"| {nome} | {cor_txt} | {inc_txt} | {acu_txt} |")

    add(f"| **Total (acurácia geral)** | | | **{_pct(tot_a, tot_v):.1f}%** |")
    add("")

    # ─── Vídeos errados (lista curta de transparência) ───────────────
    errados = [e for g_ in eficacia for e in g_["videos_errados"]]
    if errados:
        add("**Vídeos em que o sistema errou o veredito:**")
        add("")
        for e in errados:
            add(f"- `{e['video']}` — esperado `{e['esperado']}`, sistema disse `{e['veredito']}`")
        add("")

    return "\n".join(linhas) + "\n"


# ─── Orquestração ─────────────────────────────────────────────────────


def avaliar(base_videos: Path, saida_dir: Path):
    print("carregando modelos...")
    detector = ClassificadorLSTM(caminho_detector())
    avaliadores = {
        eid: ClassificadorLSTM(caminho_modelo(obter_exercicio(eid).arquivo_modelo))
        for eid in EXERCICIOS_IDS
    }

    print("carregando extrator MediaPipe...")
    extrator = ExtratorKeypoints(caminho_pose_landmarker())

    registros = []          # uma entrada por janela inferida
    videos_processados = 0
    videos_sem_janela = 0

    try:
        for video_path, pasta_exercicio, classe in descobrir_videos(base_videos):
            rel = video_path.relative_to(base_videos)
            print(f"  {rel}", end="... ", flush=True)
            janelas = coletar_janelas(extrator, video_path)
            if not janelas:
                print("(sem janelas)")
                videos_sem_janela += 1
                continue
            videos_processados += 1
            print(f"{len(janelas)} janelas")

            verdade_eh_exercicio = (pasta_exercicio != "outro")
            verdade_eh_incorreto = (classe == "incorreto") if classe else None
            video_rel = str(rel).replace("\\", "/")

            for j_idx, janela in enumerate(janelas):
                sig_det = float(detector.prever(janela))
                registro = {
                    "video": video_rel,
                    "exercicio_pasta": pasta_exercicio,
                    "classe": classe,
                    "verdade_eh_exercicio": verdade_eh_exercicio,
                    "verdade_eh_incorreto": verdade_eh_incorreto,
                    "indice_janela": j_idx,
                    "sigmoid_detector": sig_det,
                    "previsto_detector": "outro" if sig_det > UMBRAL else "exercicio",
                    "sigmoid_avaliador": None,
                    "previsto_avaliador": None,
                }
                if verdade_eh_exercicio:
                    sig_av = float(avaliadores[pasta_exercicio].prever(janela))
                    registro["sigmoid_avaliador"] = sig_av
                    registro["previsto_avaliador"] = (
                        "incorreto" if sig_av > UMBRAL else "correto"
                    )
                registros.append(registro)
    finally:
        extrator.fechar()

    if not registros:
        print("\nnenhuma janela coletada — sem nada para reportar.")
        print("Verifique se há vídeos válidos em " + str(base_videos))
        return

    print(f"\ncomputando métricas sobre {len(registros)} janelas...")
    relatorio = computar_relatorio(registros, videos_processados, videos_sem_janela)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    saida_dir.mkdir(parents=True, exist_ok=True)

    json_path = saida_dir / f"{timestamp}.json"
    json_path.write_text(
        json.dumps(
            {"registros": registros, "relatorio": relatorio},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md_path = saida_dir / f"{timestamp}.md"
    md_path.write_text(gerar_markdown(relatorio, timestamp), encoding="utf-8")

    print(f"\nrelatorios gerados:")
    print(f"  {json_path}")
    print(f"  {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Avaliação em lote do pipeline.")
    parser.add_argument(
        "--videos", default=str(ROOT / "testes" / "videos"),
        help="Pasta raiz com vídeos rotulados (default: testes/videos)",
    )
    parser.add_argument(
        "--saida", default=str(ROOT / "testes" / "relatorios"),
        help="Pasta de saída para JSON + Markdown (default: testes/relatorios)",
    )
    args = parser.parse_args()

    base_videos = Path(args.videos).resolve()
    saida_dir = Path(args.saida).resolve()

    if not base_videos.exists():
        print(f"erro: pasta de vídeos não existe: {base_videos}")
        sys.exit(1)

    print(f"pasta de vídeos: {base_videos}")
    print(f"pasta de saída : {saida_dir}\n")
    avaliar(base_videos, saida_dir)


if __name__ == "__main__":
    main()
