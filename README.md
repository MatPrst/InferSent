# InferSent

This repository is a replication of the some of the results obtained by [Conneau et al. (2017)](https://arxiv.org/abs/1705.02364). It was part of the course Advanced Topics in Computational Semantics ([see](https://cl-illc.github.io/semantics-2021/)) at the University of Amsterdam.

## Dependencies

First, you will need to have [anaconda](https://docs.anaconda.com/anaconda/install/linux/) installed. Afterwards, you can install all the dependencies use the [environment file](env.yml).

```bash
conda env create -f env.yml
```

## Structure of the repository

    .
    ├── dataset.py              # Pytorch Lightning Data Module to load the SNLI dataset
    ├── env.yml                 # conda environment file
    ├── models.py               # Pytorch Lightning Module and encoders
    ├── training.py             # Main training script
    ├── utils.py                
    ├── load.py                 # Evaluate on SentEval benchmark
    └── README.md

## Training

To train the models from scratch, you can use [training.py](training.py)

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
