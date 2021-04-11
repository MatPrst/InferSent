import pytorch_lightning as pl
import torchtext
import torch
import torch.nn as nn
import argparse

from dataset import SNLIDataModule
from models import InferSent, AWEModel, LSTMModel, BiLSTMModel, MaxBiLSTMModel
from utils import EarlyStoppingLR
from pytorch_lightning.callbacks import ModelCheckpoint

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
    early_stop_lr = EarlyStoppingLR(min_lr=1e-5)



    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        filename='checkpoint-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )

    logger = pl.loggers.TensorBoardLogger(
                save_dir='./TBlogger',
                name=config.encoder
            )

    trainer = pl.Trainer(
        gpus=1 if config.cuda else 0, 
        max_epochs=config.max_epochs, 
        callbacks=[early_stop_lr, checkpoint_callback],
        limit_train_batches=0.01 if config.debug else 1.0,
        default_root_dir="./testing",
        logger=logger)

    encoder = get_encoder(config)

    model = InferSent(data_module.glove_embeddings(), encoder, config)
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)

