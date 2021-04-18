import pytorch_lightning as pl
import os
import argparse
from dataset import SNLIDataModule
from models import InferSent, AWEModel, LSTMModel, BiLSTMModel, MaxBiLSTMModel
from torchtext.data.utils import get_tokenizer
import torch

class EarlyStoppingLR(pl.callbacks.base.Callback):
    def __init__(self, min_lr):
        self.min_lr = min_lr
    
    def on_epoch_start(self, trainer, pl_module):
        current_lr = trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']
        if self.min_lr > current_lr:
            trainer.should_stop = True

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

class InferSentPrediction:
    def __init__(self, encoder):
        self.encoder = encoder
        self.model, self.data_module = self._get_pretrained_model()
        self.itoc = {
            0: "entailment",
            1: "contradiction",
            2: "neutral"
        }
    
    def _get_checkpoint(self):
        return os.path.join(os.getcwd(), "pretrained", f"{self.encoder}.ckpt")

    def _get_pretrained_model(self):
        config = {
            "checkpoint": self._get_checkpoint(),
            "seed": 42,
            "data_dir": "./data",
            "glove_max_vectors": 10000,
            "glove_dim": 300,
            "batch_size": 64,
            "cuda": True,
            "encoder": self.encoder,
            "max_epochs": None,
            "lstm_hidden_dim": 2048 if self.encoder != "awe" else 300,
            "classifier_hidden_dim": 512,
            "debug": False
        }
        config = argparse.Namespace(**config)
        pl.seed_everything(config.seed)
        data_module = SNLIDataModule(config)
        data_module.setup()
        encoder = get_encoder(config)
        model = InferSent.load_from_checkpoint(config.checkpoint, embeddings=data_module.glove_embeddings(), encoder=encoder, config=config)
        model.to("cuda")
        return model, data_module

    def _preprocess(self, sentence):
        sentence = sentence.lower()
        tokenizer = get_tokenizer("toktok")
        tok_sent = tokenizer(sentence)
        id_sent = [[self.data_module.text_field.vocab.stoi[token] for token in tok_sent]]
        length = [len(id_sent)]
        return torch.IntTensor(id_sent).to("cuda"), torch.IntTensor(length).to("cuda")

    def _combine_repr(self, premise, hypothesis):
        combination_repr = torch.cat(
            (
                premise, 
                hypothesis, 
                torch.abs(premise - hypothesis), 
                premise * hypothesis
            ),
            dim=1
        )
        return combination_repr

    def predict(self, premise, hypothesis):
        premise_tok = self._preprocess(premise)
        hypothesis_tok = self._preprocess(hypothesis)
        
        premise_repr = self.model(premise_tok)
        hypothesis_repr = self.model(hypothesis_tok)
        
        comb_repr = self._combine_repr(premise_repr, hypothesis_repr)
        prob = self.model.classifier(comb_repr)
        pred = prob.argmax(dim=-1)
        
        return self.itoc[pred.item()]
    