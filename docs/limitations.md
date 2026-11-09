# Limitações e riscos

- O desempenho depende da origem, diversidade e qualidade das imagens anotadas.
- Um detector treinado em um contexto pode não generalizar para outras câmeras, regiões, clima ou iluminação.
- mAP e métricas de detecção não garantem contagem correta.
- Tracking pode perder ou trocar IDs em oclusões e tráfego denso.
- Contagem por uma única linha pressupõe uma geometria de câmera relativamente estável.
- FPS depende de hardware, resolução, batch e implementação.
- A avaliação deve incluir exemplos de falha e comparação com ground truth.
