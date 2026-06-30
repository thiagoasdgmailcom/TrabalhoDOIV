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
