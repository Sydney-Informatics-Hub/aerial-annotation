# aerial-annotation
Open source annotations tools for aerial imagery. Part of https://github.com/Sydney-Informatics-Hub/PIPE-3956-aerial-segmentation


## Input Data Format

Download the SA1 file from [Australian Bureau of Statistics](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files) or similar.

## Output Data Format

Images and annotations stored in COCO JSON format. 



## Instructions

To prepare the environment, run the following commands:

```
conda create --name aerial-annotation python=3.9

conda activate aerial-annotation

pip install -r requirements.txt

```

### Dataset

A toy dataset has been uploaded to Roboflow. It is a small subset, containing Chatswood region, available [here](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200).

There are multiple versions of this dataset. Please ignore the first version. Version 2 and later versions are the ones that are being used. The main difference of version 2 and 3 is that [version 2](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/2) contains 90 degree augmentaions, while [version 3](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/3) does not.

For implementing this in your code, you can use the following code snippet:

```python
from roboflow import Roboflow
 
rf = Roboflow(api_key= 'your_roboflow_api_key' )
workspace_name = "sih-vpfnf" 
dataset_version = 3 
project_name = "gis-hd-200x200" 
dataset_download_name = "coco-segmentation" 

project = rf.workspace(workspace_name).project(project_name)
dataset = project.version(dataset_version).download(dataset_download_name)
```

### Data Cleaning and Preparation

Building data from OSM can be cleaned and prepared for level categorisation using the following code snippet:

```
python scripts/osm_cleaner.py --osm_path /path/to/tiles/osm_building_annotations_by_10_percent_grid/ --columns /data/osm_columns.csv 
```

Where `osm_columns.csv` is a CSV file containing the columns we are interested to keep from the OSM data. OSM columns of interest are located in `data/osm_columns.csv`.



<!-- 
# Register the dataset
from detectron2.data.datasets import register_coco_instances
dataset_name = "chatswood-dataset" #@param {type:"string"}
dataset_folder = "gis-hd-200x200" #@param {type:"string"}
register_coco_instances(f"{dataset_name}_train", {}, f"{dataset_folder}/train/_annotations.coco.json", f"/content/{dataset_folder}/train/")
register_coco_instances(f"{dataset_name}_val", {}, f"{dataset_folder}/valid/_annotations.coco.json", f"/content/{dataset_folder}/valid/")
register_coco_instances(f"{dataset_name}_test", {}, f"{dataset_folder}/test/_annotations.coco.json", f"/content/{dataset_folder}/test/")

# Use the dataset
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
cfg.DATASETS.TEST = (f"{dataset_name}_test",)
# then do the other configs

### Commit rules:

In this project, `pre-commit` is being used. Hence, please make sure you have it in your
environment by installing it with `pip install pre-commit`.

Make sure to run pre-commit on each commit. You can run it before commit on all files in the
repository using `pre-commit run --all-files`. Otherwise, you can run it via `pre-commit run`
which will only run it on files staged for commit.

Alternatively, to add the hook, after installing pre-commit, run:

```
pre-commit install
```

this will run the pre-commit hooks every time you commit changes to the repository.

