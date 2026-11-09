# Metodologia

## Pergunta experimental

O fine-tuning de um detector pré-treinado com dados de tráfego melhora a detecção das quatro classes-alvo em comparação com o modelo original?

## Desenho

- Baseline: pesos pré-treinados sem fine-tuning.
- Modelo ajustado: mesmos pesos iniciais, fine-tuning no conjunto de treino.
- Validação: utilizada para acompanhar treino e selecionar configurações.
- Teste: reservado para avaliação final; não deve orientar ajustes repetidos.

## Controle de vazamento

Quando houver frames de vídeo, agrupar por vídeo/câmera/sessão e separar grupos inteiros entre treino, validação e teste. Não permitir frames da mesma sequência em splits diferentes.

## Comparação justa

Manter o mesmo conjunto de teste, resolução, limiar de confiança, hardware e procedimento de avaliação para baseline e modelo ajustado. Registrar diferenças inevitáveis.

## Métricas

Reportar precision, recall, mAP@0.5, mAP@0.5:0.95, latência/FPS e métricas de contagem comparadas com ground truth manual. Apresentar resultados por classe sempre que possível.
