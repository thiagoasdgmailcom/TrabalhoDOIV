# Trabalho DOIV - Deteccao de Falhas em Impressao 3D com YOLOv8

Projeto final da disciplina **DOIV: Introducao a Deteccao de Objetos**.

O objetivo deste trabalho e treinar um modelo de deteccao de objetos baseado em
`yolov8n.pt` para identificar pecas impressas em uma impressora 3D e possiveis
falhas visuais durante a impressao.

## Objetivo da Rede

A rede deve receber imagens ou video de uma impressora 3D e retornar bounding
boxes para a peca impressa e/ou regioes com defeito. As classes sugeridas para
um dataset especifico de impressao 3D sao:

- `printed_part`: objeto/peca em impressao;
- `spaghetti`: falha em que o filamento fica solto e embaracado;
- `warping`: deformacao ou descolamento nas bordas;
- `stringing`: fios finos entre regioes da peca;
- `layer_shift`: deslocamento entre camadas;
- `under_extrusion`: falta de material em uma regiao.

Se o dataset escolhido tiver outras classes, ajuste o arquivo
[`data/3d_printing_defects.yaml`](data/3d_printing_defects.yaml).

## Estrutura do Projeto

```text
.
+-- data/
|   +-- 3d_printing_defects.yaml    # Configuracao YOLO do dataset
|   +-- image_search_queries.yaml    # Termos para baixar imagens brutas
+-- datasets/
|   +-- README.md                   # Instrucao de organizacao do dataset
+-- reports/
|   +-- relatorio.md                # Modelo de relatorio
+-- scripts/
|   +-- check_dataset.py            # Verificacao simples do dataset YOLO
|   +-- augment_yolo_dataset.py     # Aumenta dataset YOLO com augmentations
|   +-- convert_roboflow_coco_to_yolo.py # Converte COCO/Roboflow para YOLO
|   +-- download_training_images.py  # Busca e download de imagens brutas
|   +-- download_successful_3d_prints.py # Baixa impressoes 3D bem-sucedidas
|   +-- detect_3d_printed_objects.py # Detecta apenas objetos impressos 3D
|   +-- live_detect.py              # Deteccao em tempo real pela webcam/camera
|   +-- predict.py                  # Predicao em imagens, videos ou webcam
|   +-- train.py                    # Treinamento com yolov8n.pt
|   +-- validate.py                 # Validacao/teste e metricas
|   +-- webcam_object_test.py        # Teste simples do YOLO pela webcam
+-- slides/
|   +-- apresentacao.md             # Roteiro de slides
+-- requirements.txt
```

## Preparacao do Ambiente

Crie um ambiente virtual e instale as dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

O dataset deve estar no formato YOLO:

```text
datasets/3d_printing_defects/
+-- images/
|   +-- train/
|   +-- val/
|   +-- test/
+-- labels/
    +-- train/
    +-- val/
    +-- test/
```

Cada imagem deve ter um arquivo `.txt` correspondente em `labels/`, com linhas
no formato:

```text
classe x_centro y_centro largura altura
```

As coordenadas sao normalizadas entre `0` e `1`.

Antes de treinar, valide a estrutura:

```powershell
python scripts/check_dataset.py --data data/3d_printing_defects.yaml
```

## Converter Dataset COCO do Roboflow

Se o Roboflow exportar o dataset em COCO, como `datasets/Find3dprint.coco`,
converta primeiro para YOLO:

```powershell
python scripts/convert_roboflow_coco_to_yolo.py --input datasets/Find3dprint.coco --output datasets/Find3dprint.yolo --data-yaml data/find3dprint.yaml
```

Esse comando cria:

- `datasets/Find3dprint.yolo/images/train`;
- `datasets/Find3dprint.yolo/images/val`;
- `datasets/Find3dprint.yolo/images/test`;
- `datasets/Find3dprint.yolo/labels/...`;
- `data/find3dprint.yaml`.

Depois treine com:

```powershell
python scripts/train.py --data data/find3dprint.yaml --epochs 80 --imgsz 640 --batch 8
```

## Aumentar Imagens de Treino e Teste

Para gerar mais imagens anotadas a partir do dataset YOLO convertido:

```powershell
python scripts/augment_yolo_dataset.py --dataset datasets/Find3dprint.yolo --train-copies 3 --val-copies 0 --test-copies 4 --seed 42
```

Esse comando aplica transformacoes simples, como espelhamento horizontal,
alteracao de brilho/contraste/cor e leve blur, mantendo as bounding boxes.

No dataset `Find3dprint.yolo`, o resultado atual ficou:

| Split | Antes | Depois |
|---|---:|---:|
| train | 53 | 212 |
| val | 2 | 2 |
| test | 2 | 10 |

Observacao: imagens aumentadas ajudam o treino e a demonstracao, mas nao
substituem novas imagens reais anotadas para medir generalizacao.

## Coleta de Imagens Brutas

Para montar um banco inicial de imagens por classe a partir da internet:

```powershell
python scripts/download_training_images.py --max-per-class 40
```

O script busca imagens em fontes abertas, salva em
`datasets/raw/3d_printing_defects/<classe>/` e gera `metadata.csv` com origem,
URL, autor e licenca quando disponiveis.

Para testar as buscas sem baixar:

```powershell
python scripts/download_training_images.py --dry-run --classes spaghetti warping
```

Essas imagens ainda precisam ser revisadas e anotadas com bounding boxes antes
de serem movidas para o dataset YOLO final.

## Coleta de Impressoes 3D Bem-Sucedidas

Para baixar imagens aleatorias de impressoes 3D sem falha aparente:

```powershell
python scripts/download_successful_3d_prints.py --count 100
```

As imagens serao salvas em `datasets/raw/successful_3d_prints/`, junto com
`metadata.csv` e `metadata.jsonl`.

Por padrao o script busca em DuckDuckGo, Bing, Openverse e Wikimedia:

```powershell
python scripts/download_successful_3d_prints.py --count 100 --providers duckduckgo bing openverse wikimedia
```

Para usar Google Images, configure a API oficial do Google Custom Search:

```powershell
$env:GOOGLE_CSE_API_KEY="sua_chave"
$env:GOOGLE_CSE_ID="seu_cse_id"
python scripts/download_successful_3d_prints.py --count 50 --providers google bing duckduckgo
```

O Bing usa busca HTML publica se voce nao informar chave. Se tiver uma chave da
Bing Image Search API:

```powershell
python scripts/download_successful_3d_prints.py --count 50 --providers bing --bing-api-key sua_chave
```

Tambem e possivel alterar a pasta de saida:

```powershell
python scripts/download_successful_3d_prints.py --count 50 --output datasets/raw/printed_part_ok
```

Essas imagens sao uteis para criar a classe `printed_part` ou `successful_print`,
mas ainda precisam ser revisadas e anotadas com bounding boxes.

## Treinamento

Treine a partir do modelo pre-treinado `yolov8n.pt`:

```powershell
python scripts/train.py --data data/3d_printing_defects.yaml --epochs 80 --imgsz 640 --batch 16
```

Os resultados serao salvos em `runs/detect/yolov8n_3d_print_defects/`.

Arquivos importantes gerados pelo YOLO:

- `weights/best.pt`: melhor peso do treino;
- `results.png`: curvas de treino/validacao;
- `confusion_matrix.png`: matriz de confusao;
- `val_batch*_pred.jpg`: exemplos com bounding boxes.

## Validacao e Teste

Para validar no conjunto `val`:

```powershell
python scripts/validate.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --data data/3d_printing_defects.yaml --split val
```

Para testar no conjunto `test`:

```powershell
python scripts/validate.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --data data/3d_printing_defects.yaml --split test
```

As metricas principais para o relatorio sao:

- Precision;
- Recall;
- mAP50;
- mAP50-95.

## Predicao em Imagens, Videos ou Webcam

Imagem ou pasta:

```powershell
python scripts/predict.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --source caminho/para/imagem_ou_pasta
```

Video:

```powershell
python scripts/predict.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --source caminho/para/video.mp4
```

Webcam/camera USB:

```powershell
python scripts/predict.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --source 0 --show
```

## Teste Rapido com Webcam

Para abrir a webcam e testar o modelo YOLO base:

```powershell
python scripts/webcam_object_test.py --weights yolov8n.pt
```

Esse teste usa o modelo pre-treinado no COCO. Ele detecta classes gerais como
pessoa, garrafa, copo, cadeira etc., mas nao possui a classe especifica
`printed_part`. Depois de treinar o modelo do projeto, use:

```powershell
python scripts/webcam_object_test.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --target-classes printed_part
```

Teclas durante a execucao:

- `q`: sair;
- `s`: salvar o frame atual em `runs/webcam_object_test/`.

## Deteccao Apenas de Objetos Impressos 3D

Depois de treinar um modelo com a classe `printed_part`, use:

```powershell
python scripts/detect_3d_printed_objects.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --source 0
```

Esse script filtra as deteccoes e mostra apenas classes que representam objetos
impressos em 3D. Para usar outro nome de classe:

```powershell
python scripts/detect_3d_printed_objects.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --printed-classes printed_part successful_print
```

## Deteccao em Tempo Real com Alerta

Para monitorar uma camera e destacar falhas:

```powershell
python scripts/live_detect.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --source 0
```

O script considera como falhas as classes:

```text
spaghetti, warping, stringing, layer_shift, under_extrusion
```

Voce pode alterar isso com:

```powershell
python scripts/live_detect.py --weights runs/detect/yolov8n_3d_print_defects/weights/best.pt --defect-classes spaghetti warping
```

## Entrega

Use os arquivos abaixo como base:

- [`reports/relatorio.md`](reports/relatorio.md): relatorio curto;
- [`slides/apresentacao.md`](slides/apresentacao.md): roteiro de apresentacao.

Depois do treino, copie para o relatorio:

- nome e fonte do dataset;
- classes utilizadas;
- quantidade de imagens por conjunto;
- metricas de treino/teste;
- imagens de exemplo com bounding boxes geradas em `runs/`.
