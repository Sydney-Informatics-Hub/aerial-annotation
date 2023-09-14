# aerial-annotation
Open source annotations tools for aerial imagery. Part of https://github.com/Sydney-Informatics-Hub/PIPE-3956-aerial-segmentation


## Input Data Format

Download the SA1 file from https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files or similar.


## Output Data Format

Images and annotations stored in COCO JSON format. 


## Instructions

```
conda create --name aerial-annotation python=3.9

conda activate aerial-annotation

pip install -r requirements.txt

```

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

