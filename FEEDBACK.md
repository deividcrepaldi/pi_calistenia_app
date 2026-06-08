# Feedback em tempo real — regras e tecnologias

Documento de referência do que a aplicação faz entre o frame que sai da
câmera (ou do vídeo importado) e o veredito que aparece na tela.

---

## 1. Pilha tecnológica

| Camada | Tecnologia | Função no pipeline |
|---|---|---|
| Linguagem | Python 3.11 | Compatível com TensorFlow ≤ 2.15 (3.12+ não tem wheel oficial). |
| Detecção de pose | MediaPipe Pose Landmarker (Tasks API) — `pose_landmarker_full.task` | 33 keypoints `(x, y, z, visibility)` por frame. Modo `VIDEO` com timestamps monotônicos. |
| Modelo de classificação | TensorFlow / Keras | Duas LSTMs por sessão: `detector.h5` + avaliador específico do exercício. |
| Captura | OpenCV (`cv2.VideoCapture`) | Webcam em tempo real ou arquivo `.mp4/.mov/.avi/.mkv/.webm/.m4v`. |
| Renderização do vídeo | OpenCV + Pillow + Tkinter (`PhotoImage`) | Esqueleto, banner de status e borda colorida desenhados sobre o frame BGR antes de virar `PhotoImage`. |
| Interface | Tkinter (tema escuro próprio) | Painel lateral com classificação, bloco de erro e botões. |
| Numérico | NumPy | Buffer circular `(30, 108)`, cálculo dos 9 ângulos, EMA. |

`requirements.txt`:
```
mediapipe==0.10.33
tensorflow
opencv-python
numpy
Pillow
```

---

## 2. Forma da entrada do modelo

Todo frame válido vira um vetor de **108 features**:

```
[ x0, y0, vis0, x1, y1, vis1, ..., x32, y32, vis32,    # 99 coords
  ang0, ang1, ang2, ang3, ang4, ang5, ang6, ang7, ang8 ]  # 9 ângulos
```

Os 9 ângulos articulares (graus, `[0, 180]`) saem da fórmula
`arccos(dot(BA, BC) / (|BA|·|BC| + 1e-6))` aplicada às tripletas:

| Índice | Nome | Tripleta MediaPipe (A, B, C) — ângulo em B |
|---|---|---|
| 99  | `cotovelo_esq`  | (15, 13, 11) |
| 100 | `cotovelo_dir`  | (16, 14, 12) |
| 101 | `joelho_esq`    | (23, 25, 27) |
| 102 | `joelho_dir`    | (24, 26, 28) |
| 103 | `quadril_esq`   | (11, 23, 25) |
| 104 | `quadril_dir`   | (12, 24, 26) |
| 105 | `alinhamento`   | (11, 23, 27) — tronco |
| 106 | `ombro_esq`     | (13, 11, 23) |
| 107 | `ombro_dir`     | (14, 12, 24) |

O cálculo é byte a byte idêntico ao do treinamento
(`pi_calistenia/extrair_keypoints.py`). Qualquer divergência aqui
invalida as predições.

### Buffer circular

- Tamanho: **30 frames** (≈ 1 s a 30 fps)
- Passo: **15 frames** (≈ 0,5 s entre inferências)
- Inferência só ocorre quando o buffer está cheio E 15 frames novos
  entraram desde a última inferência.
- Entrada apresentada ao modelo: `(1, 30, 108)` em `float32`.

---

## 3. Arquitetura em cascata

Resolve o problema de *out-of-distribution*: os avaliadores foram
treinados só com amostras do exercício, então uma pessoa em pé ou
sentada produz uma saída sigmoide matematicamente válida mas
semanticamente sem sentido (ex.: 99% "incorreto" pra alguém parado).

```
                  buffer (1, 30, 108)
                          │
                          ▼
                    detector.h5
                  ╱            ╲
            > 0.5                ≤ 0.5
           "outro"            exercício
          fora_posicao             │
        (avaliador NÃO              ▼
         é consultado)        <exercicio>.h5
                                ╱        ╲
                            > 0.5         ≤ 0.5
                          incorreto      correto
                              │
                              ▼
                       detector de erro
                       postural (regras)
```

| Modelo | Tarefa | Acurácia teste |
|---|---|---|
| `detector.h5`   | exercício vs outro                  | ~96.8 % |
| `push-up.h5`    | correto vs incorreto (push-up)      | 91.5 % |
| `plank.h5`      | correto vs incorreto (plank)        | 100 % |
| `sideplank.h5`  | correto vs incorreto (side plank)   | 83.1 % |

Limiares definidos em `inferencia.py`:

```python
UMBRAL_DETECTOR    = 0.5    # > limiar -> "outro"
UMBRAL_INCORRETO   = 0.60   # avaliador, voto "incorreto"
UMBRAL_CORRETO     = 0.40   # avaliador, voto "correto"
VOTOS_PARA_TROCAR  = 2      # votos consecutivos pra trocar de classe
```

---

## 4. Estabilização do feedback

Sem essas três camadas, o painel pisca várias vezes por segundo entre
estados — visualmente inutilizável durante o exercício.

### 4.1 EMA dos ângulos (exibição apenas)
- Filtro: `estado = 0.25 · atual + 0.75 · estado_anterior`
- Aplicado **somente** aos ângulos exibidos e à detecção do erro
  postural (entrada do modelo continua bruta).
- `nucleo/suavizacao.py :: SuavizadorEMA`

### 4.2 EMA dos landmarks (desenho apenas)
- Filtro: `estado = 0.30 · atual + 0.70 · estado_anterior` sobre os
  33 landmarks `(x, y, visibility)`.
- Reduz o tremor visual dos pontos sobre o vídeo.
- `nucleo/suavizacao.py :: SuavizadorLandmarks`

### 4.3 Histerese + votação no avaliador

Definida em `inferencia.py :: _aplicar_avaliador`. Pseudo-código:

```
saida = avaliador.prever(janela)

se classificacao atual é None:
    primeira inferência: limiar simples 0.5 (feedback imediato)

senão se saida está em [0.40, 0.60):
    zona cinza — mantém a classe atual, não atualiza confiança

senão se voto coincide com classe atual:
    reconfirma — atualiza apenas a confiança exibida

senão (voto na classe oposta):
    incrementa contador de votos
    troca quando contador >= VOTOS_PARA_TROCAR
```

Com inferência a cada 15 frames (~0,5 s a 30 fps), 2 votos equivalem a
~1 s de sinal forte contínuo antes de trocar de "correto" pra
"incorreto" ou vice-versa.

### 4.4 Decisão direta no detector

O detector **não** usa histerese: já opera sobre 30 frames inteiros
(1 s de sinal), o que por si só absorve picos. Adicionar histerese
aqui só atrasaria a recuperação quando o usuário voltasse à postura.

Quando o detector emite "outro" depois de uma execução, o estado do
avaliador é zerado (`classificacao = None`, `votos = 0`) — assim a
primeira predição após o usuário voltar à postura tem efeito
imediato em vez de ficar presa em voto de transição da execução
anterior.

---

## 5. Detecção do erro postural específico

Quando o avaliador classifica como **incorreto**, cada exercício
testa um conjunto de regras biomecânicas sobre os ângulos
**suavizados**. Cada regra devolve um número de graus de desvio
(quanto além do limite) ou `None`. A regra com maior desvio é a
reportada.

`nucleo/detector_erros.py :: identificar_erro` retorna
`(nome, correcao, desvio_em_graus)`.

### Push-up — `vista=lateral`, `tipo=dinâmico`

| Regra | Condição | Desvio | Correção |
|---|---|---|---|
| Quadril caindo      | `alinhamento < 160°`                                      | `160 − alinhamento`        | "Eleve o quadril para alinhar com os ombros" |
| Quadril empinando   | `alinhamento > 190°`                                      | `alinhamento − 190`        | "Abaixe o quadril para alinhar com os ombros" |
| Cotovelo muito aberto | `cotovelo_esq > 160°` E `cotovelo_dir > 160°`           | `min(cotovelos) − 160`     | "Mantenha os cotovelos próximos ao corpo" |

### Plank — `vista=lateral`, `tipo=estático`

| Regra | Condição | Desvio | Correção |
|---|---|---|---|
| Quadril caindo     | `alinhamento < 160°`                                       | `160 − alinhamento`        | "Contraia o core e eleve o quadril" |
| Quadril empinando  | `alinhamento > 190°`                                       | `alinhamento − 190`        | "Abaixe o quadril até alinhar com o tronco" |
| Tronco rodado      | `|quadril_esq − quadril_dir| > 15°`                        | `diff − 15`                | "Mantenha os quadris paralelos ao chão" |
| Cabeça baixa       | `min(ombro_esq, ombro_dir) < 50°`                          | `50 − min(ombros)`         | "Mantenha a cabeça alinhada com a coluna" |

### Side plank — `vista=frontal`, `tipo=estático`

| Regra | Condição | Desvio | Correção |
|---|---|---|---|
| Quadril caindo     | `quadril_esq < 150°` E `quadril_dir < 150°`               | `150 − min(quadris)`       | "Eleve o quadril lateralmente" |
| Corpo desalinhado  | `|ombro_esq − ombro_dir| > 20°`                            | `diff − 20`                | "Alinhe os ombros verticalmente" |

Adicionar um exercício novo é só registrar uma instância de
`Exercicio` em `config/exercicios.py` com suas próprias `RegraErro`s —
nenhum outro módulo precisa mudar.

---

## 6. Apresentação visual

### 6.1 Sobre o vídeo (`nucleo/desenho.py :: desenhar_status_overlay`)

Banner grande no topo do frame BGR + borda do frame na mesma cor.
A cor codifica o estado:

| Estado | Cor | Banner |
|---|---|---|
| Coletando frames pra primeira inferência   | Azul     | `AGUARDANDO` + `Coletando frames N/30` |
| Detector classificou como "outro"          | Cinza    | `POSIÇÃO NÃO RECONHECIDA` + confiança do detector |
| Avaliador → correto                        | Verde    | `CORRETO · NN%` + "Mantenha a postura" |
| Avaliador → incorreto                      | Vermelho | `INCORRETO · NN%` + nome do erro + `desvio de N°` |

A premissa é que durante o exercício o usuário olha pra si mesmo no
vídeo, então o feedback precisa estar **em cima da imagem**, não num
painel lateral em fonte pequena.

Tag discreta no canto inferior direito mostra a fonte em uso
(`Câmera ao vivo` ou `Vídeo importado`) — útil quando se está
testando com vídeos pré-gravados.

### 6.2 Painel lateral

- Cabeçalho: nome do exercício, vista, fonte
- Card de classificação (fonte 28pt, bold) com a confiança abaixo
- Bloco de erro (só aparece quando `classificacao = incorreto`):
  - Nome do erro
  - Magnitude do desvio em graus
  - Texto de correção
- Botões "Voltar" e "Encerrar"

A grade numérica de ângulos articulares foi removida intencionalmente
— durante a execução o usuário não consegue acompanhar números
mudando 30 vezes por segundo. A informação angular só aparece quando
é acionável: dentro do bloco de erro, na forma "desvio detectado: N°".

---

## 7. Ciclo de vida da tela de análise

```
TelaAnalise.__init__
    └── thread separada: carrega Inferencia + ExtratorKeypoints + fonte
                          └── ao terminar, chama _publicar() na main loop
                                └── _loop() recursivo via after(15ms):
                                      1. fonte.ler_frame()
                                      2. extrator.detectar(frame) -> (landmarks, coords_99)
                                      3. inferencia.processar_frame(coords_99)
                                      4. desenhar_esqueleto(frame, landmarks_suavizados)
                                      5. desenhar_status_overlay(frame, resultado)
                                      6. canvas_video.exibir_frame(frame)
                                      7. _atualizar_painel(resultado)
```

`_loop` agenda a si mesmo a cada 15 ms; a cadência real é limitada
pelo MediaPipe (~30 fps com `pose_landmarker_full`).

A separação entre thread de carregamento e main loop garante que a
janela permanece responsiva enquanto o TensorFlow inicializa.

---

## 8. Fontes de vídeo

Ambas implementam o mesmo protocolo: `iniciar() -> bool`,
`ler_frame()`, `parar()`, `erro_abertura`. A `TelaAnalise` recebe
uma fábrica e um rótulo — não sabe qual fonte está em uso.

| Classe | Origem | Característica |
|---|---|---|
| `captura.camera.CameraThread` | webcam (índice 0, 1280×720) | Lê o mais rápido possível, mantém só o frame mais recente — evita acumular atraso. |
| `captura.video_arquivo.VideoArquivoThread` | arquivo de vídeo | Respeita o FPS nativo do arquivo (sem isso passa rápido demais pra acompanhar o feedback) e dá loop ao chegar no final. |

---

## 9. O que NÃO entra no modelo

- Coordenadas e ângulos suavizados — a entrada do modelo é sempre
  bruta, idêntica byte a byte ao pipeline de treino. EMA é só pra
  exibição.
- Frames descartados pelo MediaPipe (visibilidade média < 0,5) —
  nesses frames `coords_99 = None` e o buffer simplesmente não avança.
- Heurísticas de plausibilidade biomecânica — o gate de posição é
  feito pelo `detector.h5`, não por regras manuais (a versão anterior
  baseada em proporção do bounding-box foi descartada quando o
  detector LSTM passou a existir).
