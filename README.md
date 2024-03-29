# Connectome harmonics

Github project to help us collaborate on this project in an organized way. 

## Installation
To set up and activate the required conda environment, run

```
conda env create -f environment.yaml
conda activate connectome-harmonics
```

To add this environment to jupyter, you'll need to run

```
python -m ipykernel install --user --name connectome-harmonics
```

## Usage
The organizational principle here is that each jupyter notebook should perform one analysis, or convey one idea. Code which is shared between multiple notebooks should be placed in utils, ideally also separated out into small files.

the .gitignore file will ignore anything put into `/scratchwork`, so you can put your scratch work there.
