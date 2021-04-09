import pytorch_lightning as pl
import torchtext
import torch
import torch.nn as nn
import argparse

class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.max_vectors = config.glove_max_vectors

    def setup(self, stage=None):
        self.text_field = torchtext.legacy.data.Field(lower=True, include_lengths=False, batch_first=True)
        self.label_field = torchtext.legacy.data.Field(sequential=False)

        self.snli_train, self.snli_val, self.snli_test = torchtext.legacy.datasets.SNLI.splits(self.text_field, self.label_field, root=self.data_dir)
        self.text_field.build_vocab(self.snli_train, vectors=torchtext.vocab.GloVe(name='840B', dim=300, cache=self.data_dir, max_vectors=self.max_vectors))
        self.label_field.build_vocab(self.snli_train, specials_first=False)

        self.train_iter, self.val_iter, self.test_iter = torchtext.legacy.data.BucketIterator.splits(
            (self.snli_train, self.snli_val, self.snli_test), batch_size=self.batch_size, device="cuda")
    
    def train_dataloader(self):
        return self.train_iter
    
    def val_dataloader(self):
        return self.val_iter
    
    def test_dataloader(self):
        return self.test_iter
    
    def glove_embeddings(self):
        return self.text_field.vocab.vectors

class AWEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        # B x S x 300
        return x.mean(dim=1)

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.glove_dim
        self.hidden_dim = config.lstm_hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
        )
    
    def forward(self, x):
        # B x S x 300
        x = x.permute(1, 0, 2) # S x B x 300
        h_t = torch.zeros(1, x.shape[1], self.hidden_dim).to("cuda")
        c_t = torch.zeros(1, x.shape[1], self.hidden_dim).to("cuda")
        _, (h_t, c_t) = self.lstm(x, (h_t, c_t))
        return h_t.squeeze()

class EarlyStoppingLR(pl.callbacks.base.Callback):
    def __init__(self, min_lr):
        self.min_lr = min_lr
    
    def on_epoch_start(self, trainer, pl_module):
        current_lr = trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']
        if self.min_lr > current_lr:
            trainer.should_stop = True

class InferSent(pl.LightningModule):
    def __init__(self, embeddings, encoder, config):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*config.lstm_hidden_dim, out_features=config.classifier_hidden_dim),
            nn.Linear(in_features=config.classifier_hidden_dim, out_features=3),
            nn.Softmax(dim=1)
        )
        print(self)
    
    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        premise_repr = self.forward(batch.premise)
        hypothesis_repr = self.forward(batch.hypothesis) # B x 300

        combination_repr = torch.cat(
            (
                premise_repr, 
                hypothesis_repr, 
                torch.abs(premise_repr - hypothesis_repr), 
                premise_repr * hypothesis_repr
            ),
            dim=1
        )

        out = self.classifier(combination_repr)
        loss = nn.functional.cross_entropy(out, batch.label)
        acc = (out.argmax(dim=-1) == batch.label).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        premise_repr = self.forward(batch.premise)
        hypothesis_repr = self.forward(batch.hypothesis) # B x 300

        combination_repr = torch.cat(
            (
                premise_repr, 
                hypothesis_repr, 
                torch.abs(premise_repr - hypothesis_repr), 
                premise_repr * hypothesis_repr
            ),
            dim=1
        )

        out = self.classifier(combination_repr)
        loss = nn.functional.cross_entropy(out, batch.label)
        acc = (out.argmax(dim=-1) == batch.label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
    
    def test_step(self, batch, batch_idx):
        premise_repr = self.forward(batch.premise)
        hypothesis_repr = self.forward(batch.hypothesis) # B x 300

        combination_repr = torch.cat(
            (
                premise_repr, 
                hypothesis_repr, 
                torch.abs(premise_repr - hypothesis_repr), 
                premise_repr * hypothesis_repr
            ),
            dim=1
        )

        out = self.classifier(combination_repr)
        loss = nn.functional.cross_entropy(out, batch.label)
        acc = (out.argmax(dim=-1) == batch.label).float().mean()
        self.log("test_acc", acc)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler_weight_decay = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.99, verbose=True)
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=0, verbose=True)
        return [optimizer], [scheduler_weight_decay, {
            'scheduler': scheduler_plateau, 'monitor': 'val_acc'
        }]

def get_encoder(config):
    if config.encoder == "awe":
        encoder = AWEModel
        config.lstm_hidden_dim = config.glove_dim
    elif config.encoder == "lstm":
        encoder = LSTMModel
    else:
        assert True, f"{config.encoder} encoder not supported"
    return encoder(config)

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
# parser.add_argument("--early_stopping_lr", type=float, default=1e-5, help=)

config = parser.parse_args()

pl.seed_everything(config.seed)
data_module = SNLIDataModule(config)
data_module.setup()
early_stop_lr = EarlyStoppingLR(min_lr=1e-5)
trainer = pl.Trainer(
    gpus=1 if config.cuda else 0, 
    max_epochs=config.max_epochs, 
    callbacks=[early_stop_lr],
    limit_train_batches=.05)

encoder = get_encoder(config)

model = InferSent(data_module.glove_embeddings(), encoder, config)
trainer.fit(model, data_module)
trainer.test(model, datamodule=data_module)

