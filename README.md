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

To train the models from scratch, you can use [training.py](training.py). The dataset used to train the models is the Stanford Natural Language Inference ([SNLI](https://nlp.stanford.edu/projects/snli/)) Corpus and will be downloaded to the specified directory. Additionally, the Global Vectors for Word Representation ([GloVe](https://nlp.stanford.edu/projects/glove/)) are used and will be downloaded to the same directory.

    .
    -h, --help            show this help message and exit
    --seed SEED
    --data_dir DATA_DIR   Directory where the data is stored (or will be downloaded).
    --glove_max_vectors GLOVE_MAX_VECTORS
                        Vocabulary size, if None include all words.
    --glove_dim GLOVE_DIM
                        GloVe embedding dimension.
    --batch_size BATCH_SIZE
                        Number of sentences in a single batch.
    --cuda                Run training on single GPU, if not set run on CPU.
    --encoder {awe,lstm,bilstm,bilstm-max}
                        Model of encoder to use.
    --max_epochs MAX_EPOCHS
                        Max number of epochs to train for. Training is stopped if the max
                        number of epochs is reached or ealy stopping is triggered.
    --lstm_hidden_dim LSTM_HIDDEN_DIM
                        Output dimension of the encoder. If encoder is AWE, then this will
                        be set to glove_dim.
    --classifier_hidden_dim CLASSIFIER_HIDDEN_DIM
    --debug               Only use 1% of the training data.

### Average Word Embeddings

```bash
python training.py --encoder awe --cuda
```

### LSTM

```bash
python training.py --encoder lstm --cuda
```

### biLSTM

```bash
python training.py --encoder bilstm --cuda
```

### biLSTM with max pooling

```bash
python training.py --encoder bilstm-max --cuda
```

## Pretrained models
The 4 pretrained models can be downloaded via this Google Drive [link](https://drive.google.com/drive/folders/1c8B7BmjDyPEDfZEJkIQqF3T2JPLMMlDa?usp=sharing). Download the one you are interested in and place it in the [pretrained](pretrained/) directory to be able to test it using [demo.ipynb](demo.ipynb).

## Results
You can open [demo.ipynb](demo.ipynb) to see the results and some comments about them.

| model      | dim  | SNLI-dev | SNLI-test | SentEval-micro | SentEval-macro |
|------------|------|----------|-----------|----------------|----------------|
| AWE        | 300  | 65.6     | 65.9      | 81.0           | 78.2           |
| LSTM       | 2048 | 90.4     | 79.8      | 79.2           | 76.7           |
| biLSTM     | 4096 | 78.2     | 78.6      | 80.4           | 78.3           |
| biLSTM-max | 4096 | 84.1     | 84.0      | 81.9           | 80.8           |

