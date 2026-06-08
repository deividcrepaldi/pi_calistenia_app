# Análise Biomecânica de Calistenia em Tempo Real

Aplicação desktop que identifica erros posturais em exercícios de calistenia
ao vivo, usando MediaPipe Pose Landmarker + duas LSTMs em cascata
(detector global de exercício + avaliador correto/incorreto por exercício).

Projeto Integrador — Ciência da Computação — UNESC.

---

## Repositórios

- **Aplicação:** <COLOQUE O LINK AQUI>
- **Treinamento dos modelos:** <COLOQUE O LINK DO `pi_calistenia` AQUI>

## Exercícios suportados

| Exercício | Vista | Tipo |
|---|---|---|
| Push-up | lateral | dinâmico |
| Plank | lateral | estático |
| Side Plank | frontal | estático |
| Hollow Body Hold | lateral | estático |

---

## Requisitos

- **Python 3.11** (o TensorFlow não tem wheel oficial pra 3.12+)
- Webcam funcional (ou vídeo pré-gravado para teste)
- Windows, macOS ou Linux

## Instalação

```bash
git clone <link-do-repo>
cd pi_calistenia_app

# cria o virtualenv com Python 3.11 (importante)
py -3.11 -m venv venv

# ativa o venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
```

## Executar

```bash
python app.py
```

A aplicação abre em duas telas:

1. **Seleção do exercício** — escolha o exercício e clique em "Iniciar com câmera" ou "Importar vídeo".
2. **Análise em tempo real** — vídeo da webcam ou arquivo com o esqueleto sobreposto e banner de status no topo (CORRETO / INCORRETO / POSIÇÃO NÃO RECONHECIDA).

---

## Modelos

Os arquivos `.h5` devem estar em `modelos/`:

| Arquivo | Função | Acurácia (treino) |
|---|---|---|
| `detector.h5` | exercício vs `outro` (filtro global) | ~96,8% |
| `push-up.h5` | correto vs incorreto (push-up) | ~91,5% |
| `plank.h5` | correto vs incorreto (plank) | ~100,0% |
| `sideplank.h5` | correto vs incorreto (side plank) | ~83,1% |
| `hollowbody.h5` | correto vs incorreto (hollow body) | — |

E o asset do MediaPipe em `assets/pose_landmarker_full.task` (baixar do
[Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)).

Os `.h5` são gerados pelo repositório de treinamento — não ficam versionados aqui.

---

## Base de dados

**Detector global (`detector.h5`):**
- Vídeos dos 4 exercícios (corretos e incorretos), provenientes do dataset de treinamento.
- ~900 vídeos da categoria `outro` (atividades cotidianas, pessoa em pé,
  sentada, andando), oriundos do dataset público [UCF101](https://www.crcv.ucf.edu/data/UCF101.php).

**Avaliadores binários (um por exercício):**
- Vídeos do exercício correspondente, rotulados como correto ou incorreto.
- Detalhes da coleta, anotação e divisão treino/teste estão no repositório
  de treinamento (`pi_calistenia`).

---

## Avaliação em lote (testes)

Pra medir acurácia/recall do sistema em vídeos não vistos durante o treino:

```bash
python testes/avaliar.py
```

Organize os vídeos em `testes/videos/<exercicio>/<correto|incorreto>/` e
em `testes/videos/outro/`. O script gera relatório com tabela de resultados
em `testes/relatorios/`.

Instruções completas em [`testes/README.md`](testes/README.md).

---

## Estrutura do projeto

```
pi_calistenia_app/
├── app.py                  ponto de entrada
├── inferencia.py           cascata detector + avaliador
├── requirements.txt
├── config/                 registro de exercícios, paths, tema visual
├── nucleo/                 extração, buffer, modelo, suavização, desenho
├── interface/              telas Tkinter
├── captura/                câmera e vídeo importado (threads)
├── modelos/                .h5 dos LSTMs (não versionado)
├── assets/                 pose_landmarker_full.task (não versionado)
└── testes/                 avaliação em lote
```

## Documentação adicional

- [`FEEDBACK.md`](FEEDBACK.md) — pipeline técnico (regras, limiares, arquitetura)
- [`testes/README.md`](testes/README.md) — avaliação em lote

---

## Autor

**Deivid Crepaldi Campos** — UNESC, Ciência da Computação
