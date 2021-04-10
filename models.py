import torch
import torch.nn as nn
import pytorch_lightning as pl

class InferSent(pl.LightningModule):
    def __init__(self, embeddings, encoder, config):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*self.encoder.output_dim, out_features=config.classifier_hidden_dim),
            nn.Linear(in_features=config.classifier_hidden_dim, out_features=3),
            nn.Softmax(dim=1)
        )
        print(self)
    
    def forward(self, x):
        sequences, lengths = x
        # print(sequences)
        # print(lengths)
        sequences = self.embeddings(sequences)
        # print(sequences.size())
        x = self.encoder((sequences, lengths))
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

class AWEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.glove_dim = config.glove_dim
    
    def forward(self, x):
        sequences, lengths = x
        batch_size, _, _ = sequences.size()
        # print(sequences.size())
        out = torch.zeros((batch_size, self.glove_dim), device=sequences.device)
        # print("out size=", out.size())
        for i, (seq, length) in enumerate(zip(sequences, lengths)):
            # print(seq.size())
            # print(seq[:length].size())
            # print("mean dim=", seq[:length].mean(dim=0).size())
            out[i,:] = seq[:length].mean(dim=0)
            # print(out[i])


        # batch x seq x 300
        # print(out.size())
        return out
        return x.mean(dim=1)
    
    @property
    def output_dim(self):
        return self.glove_dim

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
        sequences, lengths = x
        batch_size, _, _ = sequences.size()
        # print(sequences.size())
        # print(lengths.size())
        sequences = sequences.permute(1, 0, 2)
        pack = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.to("cpu"), enforce_sorted=False)
        # print(pack)
        # print(pack.size())
        # batch x seq x hidden_dim
        # x = x.permute(1, 0, 2) # batch x seq x hidden_dim
        # h_t = torch.zeros(1, x.shape[1], self.hidden_dim).to("cuda")
        # c_t = torch.zeros(1, x.shape[1], self.hidden_dim).to("cuda")
        h_t = torch.zeros(1, batch_size, self.hidden_dim).to("cuda")
        c_t = torch.zeros(1, batch_size, self.hidden_dim).to("cuda")
        _, (h_t, c_t) = self.lstm(sequences, (h_t, c_t))
        return h_t.squeeze()
    
    @property
    def output_dim(self):
        return self.hidden_dim

class BiLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.glove_dim
        self.hidden_dim = config.lstm_hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True
        )
    
    def forward(self, x):
        # batch x seq x hidden_dim
        # batch_size, _, _ = x.size()
        # x = x.permute(1, 0, 2) # batch x seq x hidden_dim
        # h_t = torch.zeros(2, x.shape[1], self.hidden_dim).to("cuda")
        # c_t = torch.zeros(2, x.shape[1], self.hidden_dim).to("cuda")
        # _, (h_t, c_t) = self.lstm(x, (h_t, c_t))

        sequences, lengths = x
        batch_size, _, _ = sequences.size()

        sequences = sequences.permute(1, 0, 2)
        pack = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.to("cpu"), enforce_sorted=False)

        h_t = torch.zeros(2, batch_size, self.hidden_dim).to("cuda")
        c_t = torch.zeros(2, batch_size, self.hidden_dim).to("cuda")
        _, (h_t, c_t) = self.lstm(sequences, (h_t, c_t))

        out = h_t.permute(1, 0, 2).reshape(batch_size, -1)
        return out
    
    @property
    def output_dim(self):
        return 2 * self.hidden_dim

class MaxBiLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.glove_dim
        self.hidden_dim = config.lstm_hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True
        )
    
    def forward(self, x):
        # batch x seq x hidden_dim
        # batch_size, _, _ = x.size()
        # x = x.permute(1, 0, 2) # batch x seq x hidden_dim
        # h_t = torch.zeros(2, x.shape[1], self.hidden_dim).to("cuda")
        # c_t = torch.zeros(2, x.shape[1], self.hidden_dim).to("cuda")
        # h, (h_t, c_t) = self.lstm(x, (h_t, c_t))
        # out, _ = torch.max(h.permute(1, 0, 2), dim=1)
        sequences, lengths = x
        batch_size, _, _ = sequences.size()

        sequences = sequences.permute(1, 0, 2)
        pack = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.to("cpu"), enforce_sorted=False)

        h_t = torch.zeros(2, batch_size, self.hidden_dim).to("cuda")
        c_t = torch.zeros(2, batch_size, self.hidden_dim).to("cuda")
        h, (h_t, c_t) = self.lstm(sequences, (h_t, c_t))
        out, _ = torch.max(h.permute(1, 0, 2), dim=1)
        return out
    
    @property
    def output_dim(self):
        return 2 * self.hidden_dim