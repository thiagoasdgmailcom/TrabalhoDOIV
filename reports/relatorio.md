# Relatorio - Deteccao de Falhas em Impressao 3D com YOLOv8n

## 1. Dataset Selecionado

**Nome do dataset:** preencher com o nome/fonte do dataset utilizado.

**Fonte:** preencher com link do Kaggle, Roboflow, GitHub ou dataset proprio.

**Objetivo:** detectar a peca impressa e/ou falhas visuais durante a impressao
3D, permitindo um monitoramento em tempo real do processo de fabricacao.

**Classes utilizadas:**

| ID | Classe | Descricao |
|---:|---|---|
| 0 | printed_part | Regiao principal da peca em impressao |
| 1 | spaghetti | Filamento solto/embaracado por falha de adesao ou impressao no ar |
| 2 | warping | Deformacao ou descolamento da base/bordas |
| 3 | stringing | Fios finos de material entre partes da peca |
| 4 | layer_shift | Deslocamento entre camadas |
| 5 | under_extrusion | Falta de material em uma regiao |

**Divisao dos dados:**

| Conjunto | Quantidade de imagens |
|---|---:|
| Treino | preencher |
| Validacao | preencher |
| Teste | preencher |

## 2. Rede de Deteccao Utilizada

Foi utilizado o modelo **YOLOv8n**, uma versao compacta da familia YOLOv8. O
YOLO realiza deteccao de objetos em uma unica etapa, estimando diretamente as
classes e as coordenadas das bounding boxes. A versao `n` foi escolhida por ser
leve, permitindo inferencia mais rapida e uso em cenarios de tempo real, como o
monitoramento de uma impressora 3D por webcam.

O treinamento foi iniciado a partir dos pesos pre-treinados `yolov8n.pt`,
aproveitando o aprendizado previo em imagens gerais e ajustando a rede para as
classes especificas do problema de impressao 3D.

**Parametros principais do treino:**

| Parametro | Valor |
|---|---:|
| Modelo base | yolov8n.pt |
| Tamanho da imagem | 640 |
| Epochs | 80 |
| Batch size | 16 |
| Otimizacao | padrao Ultralytics YOLO |

## 3. Metricas e Resultados

As metricas utilizadas para avaliar o modelo foram:

- **Precision:** proporcao de deteccoes corretas entre todas as deteccoes feitas;
- **Recall:** proporcao de objetos reais que foram encontrados pelo modelo;
- **mAP50:** media da precisao considerando IoU de 0,50;
- **mAP50-95:** media da precisao em varios limiares de IoU, de 0,50 a 0,95.

**Resultados obtidos:**

| Conjunto | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Validacao | preencher | preencher | preencher | preencher |
| Teste | preencher | preencher | preencher | preencher |

## 4. Exemplos de Deteccao

Inserir imagens geradas pelo YOLO com bounding boxes, por exemplo:

- `runs/detect/yolov8n_3d_print_defects/val_batch0_pred.jpg`;
- `runs/detect/predict/*.jpg`;
- frames do video em tempo real.

## 5. Conclusao

O modelo YOLOv8n demonstrou ser uma alternativa adequada para deteccao visual em
tempo real por ser leve e rapido. Para o problema de impressao 3D, o desempenho
depende diretamente da qualidade do dataset, da diversidade de angulos de camera,
iluminacao, tipos de filamento e variedade de falhas anotadas. Como melhorias,
seria possivel ampliar o dataset, balancear as classes com poucas amostras e
testar modelos maiores, como YOLOv8s ou YOLOv8m, caso o hardware permita.
