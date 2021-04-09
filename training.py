import pytorch_lightning as pl
import torchtext
import torch
import torch.nn as nn

class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, max_vectors=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_vectors=max_vectors

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
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # B x S x 300
        return x.mean(dim=1)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 300
        self.hidden_dim = 300
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=300,
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

class InferenceClassifier(pl.LightningModule):
    def __init__(self, embeddings, encoder=None, freeze=True):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*300, out_features=512),
            nn.Linear(in_features=512, out_features=3),
            nn.Softmax(dim=1)
        )
    
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

pl.seed_everything(42)
data_module = SNLIDataModule(data_dir="./data", max_vectors=10000)
data_module.setup()
early_stop_lr = EarlyStoppingLR(min_lr=1e-5)
trainer = pl.Trainer(
    gpus=1, 
    max_epochs=10, 
    callbacks=[early_stop_lr],
    limit_train_batches=1.0)

encoder = AWEModel()
# encoder = LSTMModel()

model = InferenceClassifier(data_module.glove_embeddings(), encoder)
trainer.fit(model, data_module)
trainer.test(model, datamodule=data_module)

