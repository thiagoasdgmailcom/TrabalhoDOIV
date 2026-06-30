# Apresentacao - Deteccao de Falhas em Impressao 3D com YOLOv8n

## Slide 1 - Titulo

**Deteccao de Falhas em Impressao 3D com YOLOv8n**

Nome do aluno, disciplina DOIV, MBA em Visao Computacional.

## Slide 2 - Problema

- Impressoras 3D podem apresentar falhas durante a fabricacao.
- Falhas comuns: spaghetti, warping, stringing, deslocamento de camada e falta
  de extrusao.
- Objetivo: detectar visualmente a peca e as regioes com defeito em tempo real.

## Slide 3 - Dataset

- Fonte: preencher com Kaggle, Roboflow, GitHub ou dataset proprio.
- Formato: YOLO, com imagens e arquivos `.txt` contendo bounding boxes.
- Divisao: treino, validacao e teste.
- Classes: preencher com as classes reais do dataset.

## Slide 4 - Anotacao

- Imagens anotadas com bounding boxes.
- Cada anotacao contem classe, centro da caixa, largura e altura normalizados.
- Se usado Roboflow: exportacao no formato YOLOv8.

## Slide 5 - Modelo

- Modelo utilizado: `yolov8n.pt`.
- Rede de deteccao em uma etapa.
- Escolha do YOLOv8n: modelo leve, rapido e adequado para inferencia em tempo
  real.
- Transfer learning a partir de pesos pre-treinados.

## Slide 6 - Treinamento

- Tamanho de entrada: 640 px.
- Epochs: preencher.
- Batch size: preencher.
- Comando usado:

```powershell
python scripts/train.py --data data/3d_printing_defects.yaml --epochs 80 --imgsz 640 --batch 16
```

## Slide 7 - Metricas

| Metrica | Resultado |
|---|---:|
| Precision | preencher |
| Recall | preencher |
| mAP50 | preencher |
| mAP50-95 | preencher |

## Slide 8 - Resultados Visuais

Inserir imagens com bounding boxes geradas pelo YOLO.

Sugestoes:

- matriz de confusao;
- curva de resultados;
- exemplos de predicao correta;
- exemplos de erro ou baixa confianca.

## Slide 9 - Sistema em Tempo Real

- Entrada: webcam ou camera apontada para a impressora 3D.
- Saida: bounding boxes e alerta quando uma classe de falha aparece por varios
  frames consecutivos.
- Script:

```powershell
python scripts/live_detect.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --source 0
```

## Slide 10 - Conclusao

- YOLOv8n e adequado para prototipar deteccao em tempo real.
- A qualidade das anotacoes e a diversidade do dataset sao decisivas.
- Proximos passos: aumentar dataset, coletar mais falhas reais e testar modelos
  maiores se houver GPU disponivel.
