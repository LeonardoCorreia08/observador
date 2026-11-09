# Arquitetura

## Pipeline

1. Entrada: imagem ou vídeo.
2. Pré-processamento: leitura e redimensionamento realizado pelo pipeline do detector.
3. Detecção: YOLO inicializado com pesos pré-treinados e ajustado ao dataset de tráfego.
4. Rastreamento: ByteTrack associa detecções entre frames e mantém IDs temporais.
5. Contagem: `LineCounter` contabiliza cada ID na primeira passagem pela linha virtual.
6. Saída: vídeo anotado e contagem total.

## Limites da implementação inicial

- A contagem é global, não separada por classe.
- A lógica conta o primeiro cruzamento por ID; não conta novamente se o veículo voltar a cruzar.
- O ponto usado para cruzamento é o centro vertical da caixa delimitadora. Em câmeras inclinadas, pode ser melhor usar o centro inferior da caixa ou uma região/polígono.
- O tracker pode trocar ou perder IDs; a contagem precisa ser validada contra anotação manual.
