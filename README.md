# TrafficVision AI

Sistema de visão computacional para **detecção, classificação, rastreamento e contagem de veículos** em imagens e vídeos de tráfego rodoviário, utilizando Transfer Learning.

> **Estado do projeto:** estrutura inicial / protótipo em desenvolvimento. Nenhum resultado experimental é pré-preenchido. Métricas, pesos e conclusões só devem ser adicionados após execução real.

## Objetivos

- Adaptar um detector pré-treinado ao domínio de tráfego rodoviário.
- Comparar o modelo original (baseline) com o modelo após fine-tuning.
- Detectar quatro categorias: carro, motocicleta, ônibus e caminhão.
- Rastrear veículos entre frames e evitar contagens duplicadas.
- Avaliar qualidade, velocidade e limitações do sistema.
- Manter experimentos reproduzíveis.

## Abordagem

1. **Baseline:** executar um modelo YOLO pré-treinado sem fine-tuning.
2. **Transfer Learning:** ajustar o modelo com um dataset anotado de tráfego.
3. **Tracking:** associar detecções entre frames com ByteTrack.
4. **Counting:** contar cada ID uma única vez ao cruzar uma linha virtual.
5. **Evaluation:** avaliar detecção e contagem em dados não utilizados no treino.

A implementação inicial usa Ultralytics YOLO. O rastreamento usa o tracker ByteTrack integrado à biblioteca. As classes e os IDs devem ser conferidos com o arquivo de configuração do dataset.

## Classes do projeto

| Classe interna | Nome esperado no dataset |
|---|---|
| `car` | carro / car |
| `motorcycle` | motocicleta / motorcycle |
| `bus` | ônibus / bus |
| `truck` | caminhão / truck |

A conversão dos nomes originais para os nomes internos deve ser feita na preparação do dataset. Não misture categorias incompatíveis sem documentar a decisão.

## Estrutura

```text
TrafficVision-AI/
├── configs/
│   ├── dataset.example.yaml
│   └── training.example.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── data_card.md
│   ├── experiment_logs.md
│   ├── methodology.md
│   └── limitations.md
├── experiments/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── README.md
├── results/
│   ├── metrics/.gitkeep
│   ├── predictions/.gitkeep
│   └── README.md
├── scripts/
│   ├── evaluate.py
│   ├── predict.py
│   ├── track_count.py
│   └── train.py
├── src/trafficvision/
│   ├── __init__.py
│   ├── config.py
│   └── counting.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10 ou 3.11 (recomendado para esta estrutura inicial).
- PyTorch compatível com o hardware e a versão instalada do Ultralytics.
- GPU NVIDIA é recomendada para treinamento, mas não obrigatória para inferência.
- Dataset com imagens e anotações no formato YOLO.

Instalação:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para ambientes com GPU, instale a versão do PyTorch apropriada para sua versão de CUDA seguindo a página oficial do PyTorch antes de instalar/ajustar o restante das dependências.

## Dataset

O projeto não inclui imagens ou rótulos de terceiros. Escolha um dataset adequado, leia os termos de uso e registre a versão e a origem em `docs/data_card.md`.

O formato esperado pelo YOLO:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Cada imagem deve possuir um `.txt` correspondente. Cada linha do rótulo deve seguir:

```text
class_id x_center y_center width height
```

As coordenadas devem estar normalizadas entre 0 e 1. Os IDs das classes precisam corresponder exatamente à ordem declarada no YAML.

**Importante sobre vídeos:** se o dataset for extraído de vídeos, separe treino, validação e teste por vídeo ou sequência, nunca distribuindo frames vizinhos aleatoriamente entre os conjuntos. Isso reduz o risco de data leakage.

Copie `configs/dataset.example.yaml` para `configs/dataset.yaml` e ajuste os caminhos locais. O arquivo local `dataset.yaml` está no `.gitignore` para evitar publicar caminhos privados.

## Execução

### 1. Baseline / inferência com pesos pré-treinados

```bash
python scripts/predict.py --weights yolo11n.pt --source caminho/para/video.mp4
```

### 2. Fine-tuning

```bash
python scripts/train.py --data configs/dataset.yaml --weights yolo11n.pt --epochs 100 --imgsz 640 --batch 16
```

Ajuste `--batch` conforme a memória disponível. O comando usa pesos pré-treinados e inicia o treinamento no dataset indicado.

### 3. Avaliação

```bash
python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --data configs/dataset.yaml
```

### 4. Rastreamento e contagem

```bash
python scripts/track_count.py --weights runs/detect/train/weights/best.pt --source caminho/para/video.mp4 --line 0.5
```

`--line 0.5` define a linha horizontal no meio da imagem, em coordenada relativa (50% da altura). A contagem é incrementada quando um ID cruza essa linha na direção configurada.

## Métricas a reportar

- Precision, Recall e mAP@0.5.
- mAP@0.5:0.95.
- FPS ou tempo médio por frame.
- Erro absoluto e percentual da contagem, comparando com uma contagem manual de referência.
- Resultados por classe e por condição (dia/noite, oclusão, distância), quando houver amostras suficientes.

Não use apenas accuracy para avaliar detecção. Registe também o limiar de confiança, IoU, resolução, hardware e versão do modelo.

## Reprodutibilidade

Para cada execução, guardar:

- commit Git do código;
- versão/hash do dataset;
- versão do Python, PyTorch e Ultralytics;
- modelo inicial e origem dos pesos;
- seed, épocas, batch, resolução e learning rate;
- hardware e configurações;
- métricas e logs;
- pesos finais ou referência ao armazenamento externo.

Use `docs/experiment_logs.md` como modelo de registo. Não preencha números estimados.

## Limitações conhecidas

- A contagem por cruzamento de linha depende da qualidade do tracking.
- Oclusões, veículos muito próximos, mudanças bruscas de câmera e baixa iluminação podem causar perda ou troca de IDs.
- Um bom mAP não garante contagem correta.
- O tracker não substitui uma avaliação de contagem com ground truth.
- A configuração inicial é um ponto de partida e precisa ser validada no dataset escolhido.

## Ética, privacidade e licenças

Use apenas imagens e vídeos que tenha autorização para processar. Evite publicar rostos, matrículas, dados pessoais ou vídeos de câmeras privadas. Verifique a licença do dataset, dos pesos e das bibliotecas antes de redistribuir qualquer artefato.

## Licença

Defina uma licença para este repositório depois de confirmar os direitos sobre o código, os dados e os pesos utilizados. A licença do código não concede automaticamente direitos sobre datasets ou modelos de terceiros.
