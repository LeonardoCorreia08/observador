#  Observador - Módulo de Experimentos

Este diretório contém a suíte de experimentos do projeto **Observador**. Os scripts são projetados para validar datasets, executar inferências de detecção de objetos e realizar rastreamento (tracking) utilizando modelos YOLO e o rastreador ByteTrack.

Todos os resultados estruturados, vídeos de saída e relatórios em JSON/CSV são salvos automaticamente no diretório raiz na pasta `results/`.

---

##  Experimentos Disponíveis

### EXP-001: Validação do Dataset COCO
Este experimento realiza a leitura e validação do dataset COCO, assegurando que as anotações e categorias estão corretas e acessíveis para os próximos passos.

*   **Módulo:** `experiments.exp_001_detection.validate_coco`
*   **Dataset Utilizado:** COCO Validation
*   **Métricas de Validação:**
    *   Imagens encontradas/anotadas: 5.000
    *   Anotações de objetos: 36.781
    *   Categorias: 80 (ex: person, bicycle, car, etc.)
*   **Artefatos Gerados:** `validation_report.json`

**Como executar:**
```bash
python -m experiments.exp_001_detection.validate_coco
```

### EXP-002: Detecção de Objetos (Object Detection)
Realiza a inferência de detecção de objetos em um lote de imagens amostrais utilizando o modelo YOLO. 

*   **Módulo:** `experiments.exp_002_detection.run_detection`
*   **Dataset:** COCO 2017 (amostra de 20 imagens)
*   **Modelo:** `yolo26n.pt` (Resolução: 640px | Limiar de Confiança: 0.25)
*   **Hardware Base:** CPU
*   **Resultados de Desempenho:**
    *   Objetos detectados: 89 (Média de 4.45 por imagem)
    *   Tempo total de inferência: 11.60s
    *   Média de processamento: 0.58s por imagem (~1.72 FPS)
*   **Artefatos Gerados:** Relatórios detalhados na pasta do experimento.

**Como executar:**
```bash
python -m experiments.exp_002_detection.run_detection
```

### EXP-003: Rastreamento de Objetos (Object Tracking)
Aplica a detecção contínua aliada ao algoritmo ByteTrack para identificar e manter o ID (rastreio) dos objetos ao longo de uma sequência de frames em vídeo.

*   **Módulo:** `experiments.exp_003_tracking.run_tracking`
*   **Dataset:** MOT17 (Sequência: `MOT17-02-DPM`)
*   **Modelo e Rastreador:** `yolo26n.pt` + `bytetrack.yaml`
*   **Hardware Base:** CPU
*   **Resultados de Desempenho:**
    *   Frames processados: 300
    *   Total de detecções: 1.959 (Média de 6.53 por frame)
    *   IDs de rastreio únicos (Tracks): 46
    *   Tempo de processamento médio: 0.088s por frame (~6.55 FPS)
*   **Artefatos Gerados:**
    *   Vídeo com as bounding boxes: `tracking_output.mp4`
    *   Dados da inferência: `tracking_results.json`, `tracking_results.csv`
    *   Resumo: `summary.json`

**Como executar:**
```bash
python -m experiments.exp_003_tracking.run_tracking
```

---

##  Estrutura do Diretório de Resultados (Results)

Após a execução dos testes, a árvore de arquivos de resultados será gerada da seguinte maneira:

```text
D:\Github\observador\
└── results\
    ├── exp_001\
    │   └── validation_report.json
    ├── exp_002\
    │   └── (relatórios de detecção das imagens)
    └── exp_003\
        ├── tracking_output.mp4
        ├── tracking_results.csv
        ├── tracking_results.json
        └── summary.json
```

##  Dependências Relevantes
Durante os experimentos, certas dependências podem ser instaladas ou atualizadas automaticamente (como `lap>=0.5.12` para o ByteTrack). Recomenda-se manter o ambiente Python configurado corretamente de acordo com o `requirements.txt` do projeto principal.