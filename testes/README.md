# Avaliação em lote

Script que roda vídeos rotulados pelo MESMO pipeline da aplicação
(extrator MediaPipe + detector LSTM + avaliador LSTM) e gera relatório
com **acurácia, precisão, recall, F1 e matrizes de confusão**.

## Organização dos vídeos

Os rótulos vêm do caminho — não precisa de planilha nem JSON paralelo:

```
testes/videos/
├── push-up/
│   ├── correto/      *.mp4
│   └── incorreto/    *.mp4
├── plank/
│   ├── correto/
│   └── incorreto/
├── sideplank/
│   ├── correto/
│   └── incorreto/
├── hollowbody/
│   ├── correto/
│   └── incorreto/
└── outro/            *.mp4   ← vídeos que NÃO são exercício
                              (alguém em pé, andando, sentado…)
```

A pasta `outro/` mede o **detector global** isoladamente: ele deveria
classificar todas essas janelas como `outro` (disparar o estado
"POSIÇÃO NÃO RECONHECIDA"). As pastas de exercício medem o **avaliador**
do exercício correspondente, considerando apenas as janelas em que o
detector deixa passar.

Extensões suportadas: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`.

## Como rodar

```
python testes/avaliar.py
```

Opções:
```
python testes/avaliar.py --videos minha/pasta --saida outra/pasta
```

A primeira execução demora pra carregar o MediaPipe e os 5 modelos LSTM
(detector + 4 avaliadores). Depois disso a inferência é rápida —
~1 segundo por janela de 30 frames.

## O que é gerado

Cada execução cria dois arquivos em `testes/relatorios/`:

- `YYYY-MM-DD_HHMMSS.json` — dados brutos. Uma entrada por janela
  inferida com: vídeo, rótulo real, sigmoid do detector, classificação
  prevista, sigmoid do avaliador (quando aplicável). Útil pra refazer
  cálculos sem reprocessar os vídeos.

- `YYYY-MM-DD_HHMMSS.md` — relatório formatado pra colar no PI:
  - Resumo (vídeos processados, janelas totais)
  - Matriz de confusão do **detector** (por janela + por vídeo)
  - Matriz de confusão de **cada avaliador** (por janela + por vídeo)
  - Métricas em cada matriz: acurácia, precisão, recall, F1

## Por janela × por vídeo

Toda métrica é reportada nos dois níveis:

| Nível       | O que é uma "amostra" | O que mede |
|-------------|-----------------------|------------|
| Por janela  | Cada inferência (1 s ≈ 1 janela) | Acurácia "crua" do modelo — comparável à acurácia de treino |
| Por vídeo   | Cada arquivo de vídeo (voto majoritário das janelas) | Acurácia user-facing — o veredito que um humano veria para o vídeo todo |

No relatório do PI, a métrica por vídeo é a que melhor responde à
pergunta "o sistema acerta o veredito final do vídeo?". A métrica por
janela responde a "o modelo classifica corretamente cada trecho de
1 segundo?", que é o que de fato roda em tempo real.

## Convenções de classe positiva

Para precisão / recall, "positivo" é convencionado como a classe
**alerta** (o que justifica intervenção):

- Detector: positivo = `outro` (anomalia que devia disparar o aviso).
- Avaliador: positivo = `incorreto` (erro postural a ser corrigido).

Com essa convenção:
- **Precisão alta** ⇒ quando o sistema dispara alerta, geralmente está
  certo. Importante pra não chatear o usuário com falsos positivos.
- **Recall alto** ⇒ quando há algo errado, o sistema dispara. Importante
  pra não deixar um erro passar despercebido.

## Pré-requisitos

Os 5 modelos `.h5` precisam estar em `modelos/`:
```
modelos/detector.h5
modelos/push-up.h5
modelos/plank.h5
modelos/sideplank.h5
modelos/hollowbody.h5
```

E o asset do MediaPipe em `assets/pose_landmarker_full.task`.

Vídeos de "outro" podem vir de qualquer fonte — UCF101, gravações
caseiras de pessoa em pé, andando, escovando os dentes. Quanto mais
diversos, mais rigoroso o teste do detector.
