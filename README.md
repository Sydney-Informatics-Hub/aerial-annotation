# aerial-annotation
Open source annotations tools for aerial imagery. Part of https://github.com/Sydney-Informatics-Hub/PIPE-3956-aerial-segmentation


## Input Data Format

Images.

## Output Data Format

Images and annotations stored in COCO JSON format. 


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
