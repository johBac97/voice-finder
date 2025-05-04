# Voice Finder

This repo provides code to train a Voice Speaker Embedding Model. The model takes as input an audio recording an provides an embedding vector that caputures the speakers identity. A use case for such a model could be a speaker identification system or voice similarity search.

For a full report of the research into creating this model see the (report)[report.md].

## Code

The code is in the form of jupyter notebooks, they are all located in the `notebooks/` folder. The notebooks `create_dataset.ipynb` and `create_dataset_voxceleb2.ipynb` are used to preprocess and convert the raw datasets into zarr archives that are compatible with the model training and inference code. The notebook `train_model.ipynb` is used to train the model. 


## Installation

The repository uses `uv` as a package manager. Follow the [instructions]() to install it. Then run the command,

```
uv sync --dev
```

to install the dependencies. Launch a jupyter instance using the command,

```
uv run jupyter notebook --no-browser
```


