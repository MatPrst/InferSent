from __future__ import absolute_import, division, unicode_literals
import sys
import io
import numpy as np
import logging

####

import torch
from models import InferSent
import argparse
import pytorch_lightning as pl
from dataset import SNLIDataModule
from models import AWEModel, LSTMModel, BiLSTMModel, MaxBiLSTMModel
from torchtext.data.utils import get_tokenizer



def get_encoder(config):
    if config.encoder == "awe":
        encoder = AWEModel
        config.lstm_hidden_dim = config.glove_dim
    elif config.encoder == "lstm":
        encoder = LSTMModel
    elif config.encoder == "bilstm":
        encoder = BiLSTMModel
    elif config.encoder == "bilstm-max":
        encoder = MaxBiLSTMModel
    else:
        assert True, f"{config.encoder} encoder not supported"
    return encoder(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="path to the checkpoint of the model to evaluate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where the data is stored (or will be downloaded).")
    parser.add_argument("--glove_max_vectors", type=int, default=None, help="Vocabulary size, if None include all words.")
    parser.add_argument("--glove_dim", type=int, default=300, help="GloVe embedding dimension.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences in a single batch.")
    parser.add_argument("--cuda", action="store_true", help="Run training on single GPU, if not set run on CPU.")
    parser.add_argument("--encoder", type=str, default="awe", choices=["awe", "lstm", "bilstm", "bilstm-max"], help="Model of encoder to use.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max number of epochs to train for. Training is stopped if the max number of epochs is reached or ealy stopping is triggered.")
    parser.add_argument("--lstm_hidden_dim", type=int, default=2048, help="Output dimension of the encoder. If encoder is AWE, then this will be set to glove_dim.")
    parser.add_argument("--classifier_hidden_dim", type=int, default=512)
    parser.add_argument("--debug", action="store_true", help="Only use 1\% of the training data.")
    # parser.add_argument("--early_stopping_lr", type=float, default=1e-5, help=)

    config = parser.parse_args()

    pl.seed_everything(config.seed)
    data_module = SNLIDataModule(config)
    data_module.setup()

    encoder = get_encoder(config)

    model = InferSent.load_from_checkpoint(config.checkpoint, embeddings=data_module.glove_embeddings(), encoder=encoder, config=config)
    model.to("cuda")
    model.eval()

    PATH_TO_SENTEVAL = '../SentEval'
    PATH_TO_DATA = '../SentEval/data'
    sys.path.insert(0, PATH_TO_SENTEVAL)
    import senteval

    def prepare(params, samples):
        data_module = SNLIDataModule(config)
        data_module.setup()
        params.stoi = data_module.text_field.vocab.stoi
        params.tokenizer = get_tokenizer("toktok")
        return
    
    def batcher(params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        max_length = max(len(sent) for sent in batch)
        id_sentences = []
        lengths = []
        for sent in batch:
            id_sent = []
            lengths.append(len(sent))
            for word in sent:
                id_sent.append(params.stoi[word])
            
            while len(id_sent) < max_length:
                id_sent.append(1) # pad
            
            id_sentences.append(id_sent)
        id_sentences = torch.IntTensor(id_sentences).to("cuda")
        lengths = torch.IntTensor(lengths).to("cuda")

        embeddings = model((id_sentences, lengths))
        return embeddings.detach().cpu().numpy()

    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 2}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                    'tenacity': 5, 'epoch_size': 4}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC',
                      'MRPC', 'SICKRelatedness', 'SICKEntailment', 'STS14']

    results = se.eval(transfer_tasks)

    print(results)
