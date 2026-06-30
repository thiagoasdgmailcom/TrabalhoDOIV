# Dataset

Coloque aqui o dataset exportado no formato YOLO.

Estrutura esperada:

```text
datasets/3d_printing_defects/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Cada imagem em `images/<split>/` deve ter um arquivo `.txt` com o mesmo nome em
`labels/<split>/`.

Exemplo:

```text
images/train/frame_001.jpg
labels/train/frame_001.txt
```

Formato de cada linha do `.txt`:

```text
class_id x_center y_center width height
```

Todos os valores de posicao e tamanho devem estar normalizados entre `0` e `1`.

Sugestao de fluxo:

1. Capture imagens ou frames de videos da impressora 3D.
2. Anote as bounding boxes no Roboflow ou CVAT.
3. Exporte no formato YOLOv8.
4. Mova as pastas exportadas para `datasets/3d_printing_defects/`.
5. Ajuste `data/3d_printing_defects.yaml` com as classes reais.

## Dataset COCO do Roboflow

Se o dataset for exportado em COCO, converta para YOLO antes do treino:

```powershell
python scripts/convert_roboflow_coco_to_yolo.py --input datasets/Find3dprint.coco --output datasets/Find3dprint.yolo --data-yaml data/find3dprint.yaml
```

Depois use o YAML criado no treino:

```powershell
python scripts/train.py --data data/find3dprint.yaml --epochs 80 --imgsz 640 --batch 8
```

## Aumento de Dados YOLO

Para aumentar o conjunto convertido em YOLO:

```powershell
python scripts/augment_yolo_dataset.py --dataset datasets/Find3dprint.yolo --train-copies 3 --val-copies 0 --test-copies 4 --seed 42
```

Esse comando cria novas imagens com sufixo `__aug_` e cria os labels `.txt`
correspondentes. No dataset atual, o split ficou com:

```text
train: 212 imagens
val: 2 imagens
test: 10 imagens
```

## Coleta de Imagens Brutas da Internet

O script `scripts/download_training_images.py` baixa imagens brutas por classe em:

```text
datasets/raw/3d_printing_defects/<classe>/
```

Exemplo:

```powershell
python scripts/download_training_images.py --max-per-class 40
```

Para baixar apenas algumas classes:

```powershell
python scripts/download_training_images.py --classes spaghetti warping --max-per-class 50
```

As buscas ficam configuradas em `data/image_search_queries.yaml`.

Importante: essas imagens ainda nao possuem bounding boxes. Elas devem ser
revisadas, filtradas e anotadas no Roboflow, CVAT ou ferramenta equivalente
antes de serem usadas no treinamento YOLO.

## Coleta de Impressoes 3D Bem-Sucedidas

Para baixar imagens aleatorias de impressoes 3D aparentemente corretas:

```powershell
python scripts/download_successful_3d_prints.py --count 100
```

Para escolher os buscadores:

```powershell
python scripts/download_successful_3d_prints.py --count 100 --providers duckduckgo bing openverse wikimedia
```

Para incluir Google, configure a Google Custom Search API:

```powershell
$env:GOOGLE_CSE_API_KEY="sua_chave"
$env:GOOGLE_CSE_ID="seu_cse_id"
python scripts/download_successful_3d_prints.py --count 50 --providers google bing duckduckgo
```

As imagens sao salvas por padrao em:

```text
datasets/raw/successful_3d_prints/
```

Use essas imagens como base para anotar a classe `printed_part` ou
`successful_print`. Elas ainda nao estao prontas para treino ate receberem as
bounding boxes no Roboflow, CVAT ou ferramenta equivalente.
