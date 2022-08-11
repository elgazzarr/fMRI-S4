from src.models.sequence.ss.s4 import S4
from tqdm.auto import tqdm
from src.tasks.encoders import PositionalEncoder, Conv1DEncoder
from src.tasks.decoders import SequenceDecoder
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix


device = 'cuda:0'

class Conv1DEncoder(nn.Module):
    def __init__(self, n_layers, d_input, d_model, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.conv_init = Conv1D_layer(kernel_size, d_input, d_model)
        for _ in range(n_layers-1):
            self.conv_layers.append(Conv1D_SE_layer(kernel_size, d_model, d_model))


    def forward(self, x):
        h = self.conv_init(x)
        for conv_layer in self.conv_layers:
            h = conv_layer(h)
        return h

class Conv1D_layer(nn.Module):
    def __init__(self, kernel_size, d_in, d_out, pool=False, depthwise=False):
        super().__init__()

        if depthwise:
            self.conv = nn.Sequential(nn.Conv1d(d_in, d_in, kernel_size, groups=d_in), nn.Conv1d(d_in, d_out, 1))
        else:
            self.conv = nn.Conv1d(d_in, d_out, kernel_size)
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self, x):
        h = self.conv(x)
        h = self.activation(h)
        h = self.bn(h)

        return h




class mymodel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.conv_encoder = Conv1DEncoder(params['n_conv_layers'], params['d_input'], params['d_model'])
        self.lr = params['lr']
        self._initialize_state()
        self.loss = nn.CrossEntropyLoss()
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.clff = params['clf']
        for _ in range(params['n_s4_layers']):
            self.s4_layers.append(
                    S4(
                        d_model=params['d_model'],
                        l_max=params['T'],
                        bidirectional=True,
                        postact='glu',
                        dropout=0.2,
                        transposed=True,
                        channels = params['channels'],
                        mix=True
                    ))
            self.norms.append(nn.LayerNorm(params['d_model']))
            self.dropouts.append(nn.Dropout2d(0.2))
        self.classifier = nn.Conv1d(params['d_model'],2,1) if params['clf'] == "A" else nn.Sequential(nn.Dropout(0.5), nn.Linear(params['d_model'], 2))


    def _initialize_state(self):
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        device = device or batch[0].device
        self._state = self.s4_layers[0].default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError


    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """


        x = self.conv_encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            z, state = layer(z,state= self._state)
            self._state = state
            z = dropout(z)
            x = z + x
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        if self.clff == "A":
            x = F.relu(self.classifier(nn.functional.relu(x,inplace=True)))
            y = x.mean(dim=-1).squeeze(-1)
        else:
            x = x.mean(dim=-1).squeeze(-1)
            y = self.classifier(x)

        return y

    def configure_optimizers (self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.to(device)
        y = y.to(device)
        x = x.transpose(-1, -2)
        y_hat = self(x)
        loss =  F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.to(device)
        x = x.transpose(-1, -2)
        y = y.to(device)
        y_hat = self(x)
        _, preds = torch.max(y_hat, 1)
        correct = torch.sum(preds == y)
        return {'val_correct': correct, 'batch_size': y.shape[0]}

    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack([x['val_correct'].float() for x in outputs]).sum() / np.array([x['batch_size'] for x in outputs]).sum()
        self.log('val_acc', avg_acc, on_step=False, on_epoch=True)
        return {'val_acc': avg_acc}


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.to(device)
        x = x.transpose(-1, -2)
        y = y.to(device)
        y_hat = self(x)
        _, preds = torch.max(y_hat, 1)
        correct = torch.sum(preds == y)
        return {'test_correct': correct.float(), 'batch_size': y.shape[0], 'predictions': preds, 'labels':y}

    def test_epoch_end(self, outputs):
        preds = torch.stack([x['predictions'] for x in outputs]).ravel().detach().cpu().numpy()
        labels = torch.stack([x['labels'] for x in outputs]).ravel().detach().cpu().numpy()
        acc = round(balanced_accuracy_score(labels, preds),3)
        auc = round(roc_auc_score(labels, preds),3)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        spec = round(tn / (tn+fp), 3)
        sens = round(tp / (tp+fn), 3)

        avg_acc = torch.stack([x['test_correct'].float() for x in outputs]).sum() / np.array([x['batch_size'] for x in outputs]).sum()
        self.log('test_auc', auc)
        self.log('test_acc', acc)
        self.log('test_sens', sens)
        self.log('test_spec', spec)
        return {'test_auc': auc, 'test_acc': acc, 'test_sens': sens, 'test_spec': spec }
