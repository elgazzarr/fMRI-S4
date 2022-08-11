import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from datasets import Ukbb, Mddrest, Abide, get_dataset_class

import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse
import numpy as np
import wandb
from src.models.sequence.ss.s4 import S4, S4_Graph
from tqdm.auto import tqdm
from src.tasks.encoders import PositionalEncoder, Conv1DEncoder
from src.tasks.decoders import SequenceDecoder
from src.tasks.metrics import mae, mse, corr, mae_mask, mse_mask, corr_mask
import string
import random
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val




def prepare_loaders(args, dataset, task, atlas, ss_task, mask_rois, ad):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_train = transform_test = transform

    print(f'==> Preparing {dataset} data..')

    dataset_args = {"data_len" : args.l_full , "pred_len" : args.l_predict, "atlas":atlas,
    "mask_rois":mask_rois, "predict": ss_task, "label": args.label, "task": task}

    trainvalset = get_dataset_class(dataset)(data_path='csvfiles/{}_train.csv'.format(dataset), ad=ad, **dataset_args)
    trainset, valset = split_train_val(trainvalset, val_split=0.2)




    testset = get_dataset_class(dataset)(data_path='csvfiles/{}_test.csv'.format(dataset), ad=False, **dataset_args)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    return trainloader, valloader, testloader


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   #pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, in_channels, kernel_size, dilation=dilation,  **kwargs)


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        task,
        d_output=2,
        l_full = 150,
        l_output= 50,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        channels=1,
        mask_rois=[],
        gcn=True,
        predict_mode='forecast'
    ):
        super().__init__()

        self.prenorm = prenorm
        self.task = task
        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        #self.encoder = nn.Linear(d_input, d_model)
        self.encoder = CausalConv1d(d_input, d_model, 1)
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.E = nn.Parameter(torch.randn(d_model, channels), requires_grad=True)
        #self.E = nn.Parameter(torch.tensor(adj.astype(np.float32)), requires_grad=False)
        #self.e2 = nn.Parameter(torch.randn(10, d_model), requires_grad=True)

        for _ in range(n_layers):
            if gcn:
                self.s4_layers.append(
                    S4_Graph(
                        d_model=d_model,
                        l_max=l_full,
                        bidirectional=True,
                        postact='glu',
                        dropout=dropout,
                        transposed=True,
                        channels = channels,
                        use_gcn= gcn,
                        mix=True,
                        E= self.E
                    )
                )
            else:
                self.s4_layers.append(
                    S4(
                        d_model=d_model,
                        l_max=l_full,
                        bidirectional=True,
                        postact='glu',
                        dropout=dropout,
                        transposed=True,
                        channels = channels,
                        mix=True
                    ))

            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        # Linear decoder
        self.decoder_ss = SequenceDecoder(d_model, d_output, l_output, rois=mask_rois, mode=predict_mode)
        self.decoder_s = nn.Linear(d_model, 2)
        self._initialize_state()

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

        #print('')
        #print('input? ', torch.isnan(x).any())

        x = x.transpose(-1, -2)

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        #x = x[:, :, :-self.encoder.padding[0]]


        #x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            #print(z.shape)
            z, state = layer(z,state= self._state)
            #print('S4? ',  torch.isnan(z).any())
            self._state = state

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                #print('Norm? ',  torch.isnan(x).any())
        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        #x = x.mean(dim=1)

        # Decode the outputs
        if self.task == 'supervised':
            x = x.mean(dim=1)
            y = self.decoder_s(x)
        else:
            y = self.decoder_ss(x,state)  #self.decoder_ss(x,state)

        return y


def setup_optimizer(model, lr, weight_decay, patience):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in set(frozenset(hp.items()) for hp in hps)
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.7, mode='max')

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler



###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training

def train_selfsupervised(model, trainloader, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets, _) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        mse_loss = mse(outputs, targets)
        corr_loss = corr(outputs, targets)

        loss = mse_loss #2*mse_loss -0.5*corr_loss #+ mse_loss
        #loss_back = -0.5corr_loss + 2*mse_loss #(-0.5 * corr_loss)  +  2* mse_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Train Loss: %.3f' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1))
        )
    wandb.log({'Train_loss_ss': train_loss/(batch_idx+1)})



def train_supervised(model, train_loader, optimizer):

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    criteration =  nn.CrossEntropyLoss()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (inputs,_, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criteration(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Train Loss: %.3f' %
            (batch_idx, len(train_loader), train_loss/(batch_idx+1))
        )
    wandb.log({'Train_loss_sup': train_loss/(batch_idx+1)})








def eval_supervised(model,  epoch, dataloader, checkpoint_path, best_acc, es_counter = 0, checkpoint=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs,_, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader),  100.*correct/total, correct, total)
            )

    acc = 100.*correct/total
    ce_loss = eval_loss/(batch_idx+1)

    if checkpoint:
        if acc >= best_acc:
            state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_path)
            best_acc = acc
            es_counter = 0
        else:
            es_counter += 1

    return ce_loss, acc, es_counter, best_acc

def save_preds(outputs, targets, y):
    outputs = np.concatenate(outputs,axis=0)
    targets = np.concatenate(targets,axis=0)
    y = np.array(y,dtype=np.float32).ravel()
    np.savez('./outputs/pred-default.npz', output=outputs, target=targets, y = y)

def eval_anomaly(model, dataloader, mask_ind):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        out = []
        ts = []
        preds = []
        y = []
        for batch_idx, (inputs, targets, labels) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mask = torch.zeros((inputs.shape[0],112))
            mask[:,mask_ind] = 1
            corr_score = corr(outputs, targets).cpu().detach().numpy()
            mae_score = mae(outputs, targets).cpu().detach().numpy()
            #mae = torch.mean(torch.abs(torch.subtract(outputs,targets)),axis=1).cpu().detach().numpy().mean(axis=-1)
            errors = -corr_score#mae_score
            preds.append(errors)
            y.append(labels.cpu().detach().numpy())

            out.append(outputs.cpu().detach().numpy())
            ts.append(targets.cpu().detach().numpy())
        save_preds(out, ts, y)
        auc = roc_auc_score(np.array(y,dtype=np.float32).ravel(), np.array(preds).ravel())
        return auc


def eval_selfsupervised(model, epoch, dataloader,checkpoint_path, best_corr,  es_counter = 0, checkpoint=False,):

    model.eval()
    eval_loss_mse = 0
    eval_loss_mae = 0
    eval_loss_corr = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        preds = []
        labels = []
        data = []
        for batch_idx, (inputs, targets,_) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_mse = mse(outputs, targets)
            loss_mae = mae(outputs, targets)
            loss_corr = corr(outputs, targets)

            data.append(inputs.cpu().detach().numpy())

            preds.append(outputs.cpu().detach().numpy())
            labels.append(targets.cpu().detach().numpy())
            eval_loss_mse += loss_mse.item()
            eval_loss_mae += loss_mae.item()
            eval_loss_corr += loss_corr.item()
            pbar.set_description(
                'Batch Idx: (%d/%d) | MSE Loss: %.3f | MAE Loss: %.3f | CORR score : %.3f' %
                (batch_idx, len(dataloader), eval_loss_mse/(batch_idx+1), eval_loss_mae/(batch_idx+1), eval_loss_corr/(batch_idx+1))
            )

    mse_error = eval_loss_mse/(batch_idx+1)
    mae_error = eval_loss_mae/(batch_idx+1)
    corr_score  = eval_loss_corr/(batch_idx+1)
    if checkpoint:
        if corr_score > best_corr:
            state = {
                'model': model.state_dict(),
                'mse': corr_score,
                'epoch': epoch,
            }

            torch.save(state, checkpoint_path)
            best_corr = corr_score
            es_counter = 0
        else:
            es_counter += 1
    else:
        save_preds_ss(data, preds, labels)
    return mse_error, mae_error, corr_score, es_counter, best_corr

def save_preds_ss(inputs, outputs, targets):
    inputs = np.concatenate(inputs,axis=0)
    outputs = np.concatenate(outputs,axis=0)
    targets = np.concatenate(targets,axis=0)

    np.savez('./outputs/ss-losses-2.npz', input=inputs, output=outputs, target=targets)
