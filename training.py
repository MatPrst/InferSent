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
        return nn.functional.cross_entropy(out, batch.label)
    
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
        self.log("val_loss", nn.functional.cross_entropy(out, batch.label))
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


data_module = SNLIDataModule(data_dir="./data", max_vectors=10000)
data_module.setup()
trainer = pl.Trainer()

encoder = AWEModel()

model = InferenceClassifier(data_module.glove_embeddings(), encoder)
trainer.fit(model, data_module)


