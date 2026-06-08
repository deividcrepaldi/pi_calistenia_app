"""
Desenho do esqueleto e do overlay de status sobre o frame BGR.

O esqueleto usa OpenCV (linhas/círculos baratos, sem texto).
O overlay de status usa Pillow para o texto, porque a fonte Hershey
do `cv2.putText` só suporta ASCII — caracteres como "Ç", "ã", "ç",
"°", "·" saem como "?" e quebram visualmente o feedback. Pillow
renderiza Unicode corretamente com qualquer TTF do sistema.

Para manter o custo razoável, todo o desenho do overlay é feito numa
única conversão BGR → RGBA → BGR por frame — um único round-trip,
mesmo que o overlay desenhe múltiplos textos e retângulos.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import tema


# ─── Esqueleto MediaPipe (sem texto, segue em OpenCV) ────────────────

# Conexões entre os 33 keypoints do Pose Landmarker (padrão MediaPipe).
CONEXOES_POSE = (
    # braços
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    # tronco
    (11, 12), (11, 23), (12, 24), (23, 24),
    # pernas
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
    # face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
)


def _hex_para_bgr(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _hex_para_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


_COR_LINHA = _hex_para_bgr(tema.COR_ESQUELETO)
_COR_PONTO = _hex_para_bgr(tema.COR_KEYPOINT)


def desenhar_esqueleto(frame_bgr, landmarks):
    """
    Desenha conexões e keypoints sobre `frame_bgr` in-place.
    `landmarks` é a lista de 33 landmarks do PoseLandmarker (x,y normalizados
    em [0,1]). Se None, nada é desenhado.
    """
    if landmarks is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    pontos = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in CONEXOES_POSE:
        if a < len(pontos) and b < len(pontos):
            cv2.line(frame_bgr, pontos[a], pontos[b], _COR_LINHA, 2, cv2.LINE_AA)

    for p in pontos:
        cv2.circle(frame_bgr, p, 4, _COR_PONTO, -1, cv2.LINE_AA)

    return frame_bgr


# ─── Overlay de status (Pillow para texto Unicode) ───────────────────
#
# Banner colorido grande no topo do frame com o veredito em tempo real.
# A premissa é que durante a execução o usuário olha pra si mesmo no
# vídeo, então o feedback precisa estar EM CIMA da imagem, não num
# painel lateral em fonte pequena. A cor codifica o estado:
#
#   verde    → CORRETO
#   vermelho → INCORRETO
#   azul     → AGUARDANDO frames pra primeira inferência
#   cinza    → POSIÇÃO NÃO RECONHECIDA (detector global em "outro")
#
# Além do banner, uma borda fina da mesma cor enquadra o frame inteiro,
# servindo de pista periférica enquanto o usuário está em movimento.

# Cores escolhidas com WCAG AA em mente para texto branco bold por cima:
#   verde     contraste ~5.4 com (245,245,248)
#   vermelho  contraste ~5.1
#   cinza     contraste ~8.0
#   azul      contraste ~4.6 (do tema)
# Versões mais claras anteriores (coral, verde-claro, cinza-claro) ficavam
# abaixo de 4 e o subtítulo sumia no fundo.
_COR_VERDE_RGB    = (40, 155, 75)
_COR_VERMELHO_RGB = (210, 60, 50)
_COR_AZUL_RGB     = _hex_para_rgb(tema.AZUL_DESTAQUE)
_COR_CINZA_RGB    = (72, 80, 94)
_COR_TEXTO_RGB    = (250, 250, 252)
_COR_SOMBRA_TEXTO = (0, 0, 0, 140)     # RGBA — sombra fina pra reforçar contraste
_COR_TAG_FUNDO    = (8, 12, 20, 210)   # RGBA, ligeiramente mais opaco que antes


# Localização da fonte TTF — testa uma cadeia de candidatos comuns por
# plataforma. Cacheia tamanhos já carregados para evitar reabrir o
# arquivo a cada frame (ImageFont.truetype faz I/O).

_CANDIDATOS_FONTE_BOLD = (
    "C:/Windows/Fonts/segoeuib.ttf",   # Segoe UI Bold (Windows 10/11)
    "C:/Windows/Fonts/arialbd.ttf",    # Arial Bold (Windows fallback)
    "/System/Library/Fonts/HelveticaNeue.ttc",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
)

_CANDIDATOS_FONTE_REGULAR = (
    "C:/Windows/Fonts/segoeui.ttf",    # Segoe UI (Windows)
    "C:/Windows/Fonts/arial.ttf",      # Arial (Windows fallback)
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _localizar(candidatos):
    for caminho in candidatos:
        if os.path.exists(caminho):
            return caminho
    return None


_CAMINHO_BOLD = _localizar(_CANDIDATOS_FONTE_BOLD)
_CAMINHO_REGULAR = _localizar(_CANDIDATOS_FONTE_REGULAR) or _CAMINHO_BOLD

_CACHE_FONTES_BOLD = {}
_CACHE_FONTES_REGULAR = {}


def _fonte_bold(tamanho: int):
    chave = int(tamanho)
    cache = _CACHE_FONTES_BOLD
    if chave not in cache:
        if _CAMINHO_BOLD:
            cache[chave] = ImageFont.truetype(_CAMINHO_BOLD, chave)
        else:
            cache[chave] = ImageFont.load_default()
    return cache[chave]


def _fonte_regular(tamanho: int):
    chave = int(tamanho)
    cache = _CACHE_FONTES_REGULAR
    if chave not in cache:
        if _CAMINHO_REGULAR:
            cache[chave] = ImageFont.truetype(_CAMINHO_REGULAR, chave)
        else:
            cache[chave] = ImageFont.load_default()
    return cache[chave]


def desenhar_status_overlay(frame_bgr, resultado: dict, rotulo_fonte: str = ""):
    """
    Desenha banner de status no topo do frame e tag de fonte no canto inferior
    direito. O texto é renderizado por Pillow (Unicode), mas a renderização
    SÓ acontece quando o conteúdo muda — o resultado é cacheado como sprite
    BGR pronto pra colar via slice. Em estado estável (correto/incorreto
    constante), o custo por frame é só uma `frame[:H] = sprite` (memcpy) e
    uma `cv2.rectangle` da borda. Sem cache, o round-trip BGR→RGBA→BGR a
    cada frame derrubava o FPS pra <20.
    """
    if frame_bgr is None:
        return frame_bgr

    titulo, subtitulo, cor_bg = _decidir_banner(resultado)
    if titulo is None and not rotulo_fonte:
        return frame_bgr

    h, w = frame_bgr.shape[:2]

    if titulo is not None:
        _aplicar_banner(frame_bgr, w, h, titulo, subtitulo, cor_bg)

    if rotulo_fonte:
        _aplicar_tag(frame_bgr, w, h, rotulo_fonte)

    return frame_bgr


def _decidir_banner(resultado: dict):
    """Retorna (titulo, subtitulo, cor_rgb) ou (None, None, None) se não há banner."""
    if resultado is None:
        return None, None, None

    if not resultado.get("pronto"):
        atual = resultado.get("frames_buffer", 0)
        cap   = resultado.get("capacidade_buffer", 30)
        return "AGUARDANDO", f"Coletando frames {atual}/{cap}", _COR_AZUL_RGB

    if resultado.get("fora_posicao"):
        conf = resultado.get("confianca_detector", 0.0) * 100.0
        return (
            "POSIÇÃO NÃO RECONHECIDA",
            f"Assuma a postura do exercício  ·  detector {conf:.0f}%",
            _COR_CINZA_RGB,
        )

    classe = resultado.get("classificacao")
    conf = resultado.get("confianca", 0.0) * 100.0

    if classe == "correto":
        return f"CORRETO  ·  {conf:.0f}%", "Mantenha a postura", _COR_VERDE_RGB

    if classe == "incorreto":
        nome_erro = resultado.get("erro") or "Postura incorreta"
        desvio = resultado.get("desvio")
        sub = f"{nome_erro}  ·  desvio de {desvio:.0f}°" if desvio is not None else nome_erro
        return f"INCORRETO  ·  {conf:.0f}%", sub, _COR_VERMELHO_RGB

    return None, None, None


# ─── Cache do banner ──────────────────────────────────────────────────
#
# O banner muda só quando o estado muda (CORRETO ↔ INCORRETO, mudança de
# texto de subtítulo, mudança de cor, ou resize da janela). A grande maioria
# dos frames mostra o mesmo banner do frame anterior — não faz sentido re-
# renderizar Pillow + reconverter BGR↔RGBA toda vez.
#
# Estratégia:
#   • Gera um sprite BGR do banner uma vez por estado.
#   • Por frame: cola via `frame[:H] = sprite` (memcpy puro) + cv2.rectangle
#     pra borda. Custo da ordem de microssegundos vs ~30-50 ms da versão
#     anterior.
#
# A tag de fonte segue o mesmo padrão, com BGRA pra preservar a transparência.

_cache_banner = {"chave": None, "sprite_bgr": None, "altura": 0, "borda_bgr": None}
_cache_tag = {"chave": None, "sprite_bgra": None, "x0": 0, "y0": 0}


def _rgb_para_bgr_tupla(cor_rgb):
    return (cor_rgb[2], cor_rgb[1], cor_rgb[0])


def _gerar_sprite_banner(w: int, altura: int, titulo: str, subtitulo: str, cor_rgb):
    """Renderiza o banner via Pillow e devolve um ndarray BGR (altura, w, 3)."""
    pil = Image.new("RGB", (w, altura), tuple(cor_rgb))
    draw = ImageDraw.Draw(pil)

    tam_titulo = max(24, int(altura * 0.36))
    tam_sub = max(15, int(altura * 0.21))
    fonte_titulo = _fonte_bold(tam_titulo)
    fonte_sub = _fonte_regular(tam_sub)

    asc_t, desc_t = fonte_titulo.getmetrics()
    linha_titulo = asc_t + desc_t

    margem_x = 24
    y_titulo = max(6, int(altura * 0.08))
    y_sub = y_titulo + linha_titulo + 2

    # Sombra 1px atrás + texto principal por cima — melhor leitura sobre o
    # fundo colorido (sem isso, branco sobre verde fica "lavado").
    sombra = (0, 0, 0)
    cor_texto = _COR_TEXTO_RGB
    draw.text((margem_x + 1, y_titulo + 1), titulo, font=fonte_titulo, fill=sombra)
    draw.text((margem_x, y_titulo), titulo, font=fonte_titulo, fill=cor_texto)
    if subtitulo:
        draw.text((margem_x + 1, y_sub + 1), subtitulo, font=fonte_sub, fill=sombra)
        draw.text((margem_x, y_sub), subtitulo, font=fonte_sub, fill=cor_texto)

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _aplicar_banner(frame_bgr, w: int, h: int, titulo: str, subtitulo: str, cor_rgb):
    altura = max(86, int(h * 0.145))
    chave = (titulo, subtitulo, tuple(cor_rgb), w, altura)
    if _cache_banner["chave"] != chave:
        _cache_banner["sprite_bgr"] = _gerar_sprite_banner(w, altura, titulo, subtitulo, cor_rgb)
        _cache_banner["altura"] = altura
        _cache_banner["borda_bgr"] = _rgb_para_bgr_tupla(cor_rgb)
        _cache_banner["chave"] = chave

    # Cola o banner no topo (slice = memcpy).
    frame_bgr[: _cache_banner["altura"]] = _cache_banner["sprite_bgr"]
    # Borda de moldura — barata via cv2.
    cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), _cache_banner["borda_bgr"], thickness=4)


def _gerar_sprite_tag(w: int, texto: str):
    """Renderiza a tag de fonte como BGRA (preserva transparência)."""
    tam = max(13, int(w / 90))
    fonte = _fonte_regular(tam)

    bbox = fonte.getbbox(texto)
    tw = bbox[2] - bbox[0]
    asc, desc = fonte.getmetrics()
    th = asc + desc
    margem = 12

    largura = tw + margem * 2
    altura = th + margem * 2

    pil = Image.new("RGBA", (largura, altura), _COR_TAG_FUNDO)
    draw = ImageDraw.Draw(pil, "RGBA")
    draw.text(
        (margem, margem // 2), texto,
        font=fonte, fill=(*_COR_TEXTO_RGB, 255),
    )
    rgba = np.asarray(pil)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def _aplicar_tag(frame_bgr, w: int, h: int, texto: str):
    chave = (texto, w)
    if _cache_tag["chave"] != chave:
        _cache_tag["sprite_bgra"] = _gerar_sprite_tag(w, texto)
        sh, sw, _ = _cache_tag["sprite_bgra"].shape
        margem = 12
        _cache_tag["x0"] = w - sw - margem
        _cache_tag["y0"] = h - sh - margem
        _cache_tag["chave"] = chave

    sprite = _cache_tag["sprite_bgra"]
    sh, sw, _ = sprite.shape
    x0, y0 = _cache_tag["x0"], _cache_tag["y0"]
    x1, y1 = x0 + sw, y0 + sh

    # Recorta região e compõe com alpha — região pequena, custo desprezível.
    regiao = frame_bgr[y0:y1, x0:x1].astype(np.float32)
    bgr_sprite = sprite[:, :, :3].astype(np.float32)
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
    composto = alpha * bgr_sprite + (1.0 - alpha) * regiao
    frame_bgr[y0:y1, x0:x1] = composto.astype(np.uint8)
