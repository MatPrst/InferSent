import pytorch_lightning as pl
import torchtext

class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.max_vectors = config.glove_max_vectors

    def setup(self, stage=None):
        self.text_field = torchtext.legacy.data.Field(lower=True, include_lengths=True, batch_first=True)
        self.label_field = torchtext.legacy.data.Field(sequential=False)

        self.snli_train, self.snli_val, self.snli_test = torchtext.legacy.datasets.SNLI.splits(self.text_field, self.label_field, root=self.data_dir)
        self.text_field.build_vocab(self.snli_train, vectors=torchtext.vocab.GloVe(name='840B', dim=300, cache=self.data_dir, max_vectors=self.max_vectors))
        self.label_field.build_vocab(self.snli_train, specials_first=False)

        self.train_iter, self.val_iter, self.test_iter = torchtext.legacy.data.BucketIterator.splits(
            (self.snli_train, self.snli_val, self.snli_test), batch_size=self.batch_size, device="cuda", shuffle=True)
    
    def train_dataloader(self):
        return self.train_iter
    
    def val_dataloader(self):
        return self.val_iter
    
    def test_dataloader(self):
        return self.test_iter
    
    def glove_embeddings(self):
        return self.text_field.vocab.vectors