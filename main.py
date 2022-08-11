import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
from datasets import Ukbb, Mddrest, Abide, get_dataset_class
from model import mymodel
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
import os
import pickle as pkl
from attribution.mask import Mask
from attribution.perturbation import FadeMovingAverage
from utils.losses import *
from attribution.mask_group import MaskGroup
from attribution.perturbation import GaussianBlur
from baselines.explainers import FO, FP, IG, SVS
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import wandb
import shutil
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Training ')

parser.add_argument('--name', default=None, type=str, help='model name')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay')
parser.add_argument('--epochs', default=60, type=int, help='Number of training epochs')
parser.add_argument('--label', default='Diagnosis',  choices=['Sex', 'MDD', 'Diagnosis'], type=str, help= 'label to detect')

# Scheduler
parser.add_argument('--patience', default=20, type=float, help='Patience for learning rate scheduler')
# Dataset
parser.add_argument('--dataset', default='Abide', choices=['Ukbb', 'Mddrest', 'Abide', 'Jpmdd'], type=str, help='Dataset')
parser.add_argument('--atlas', default='HO_sub', type=str, help='Parcellation atlas ')
parser.add_argument('--input_len', default='200', type=int, help='length of the timeseries')

# Dataloader
parser.add_argument('--num_workers', default=5, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
# Model
parser.add_argument('--n_conv_layers', default=2, type=int, help='Number of conv layers in encoder')
parser.add_argument('--n_s4_layers', default=2, type=int, help='Number of S4 layers')
parser.add_argument('--d_model', default=256, type=int, help='Model dimension')
parser.add_argument('--clf', default='B', type=str, help='Classifier head option')
parser.add_argument('--channels', default=1, type=int, help='Number of channels at each S4 Layer')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout')

args = parser.parse_args()
wandb.init('S4Classifier_Mddrest')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(32),
    )
    return train, val


def prepare_kfold_loaders(args, train_df, test_df):


    print(f'==> Preparing {args.dataset} data..')
    trainval_dataset = get_dataset_class(args.dataset)(train_df, args.atlas, args.input_len)
    trainset, valset = split_train_val(trainval_dataset, val_split=0.2)
    nrois = trainval_dataset.nrois
    testset =  get_dataset_class(args.dataset)(test_df, args.atlas, args.input_len)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=args.num_workers, drop_last=True)

    return trainloader, valloader, testloader,nrois


def train_kfold(args):

    df = pd.read_csv(f'./csvfiles/{args.dataset}.csv')
    skf = model_selection.StratifiedKFold(n_splits=5, random_state=69, shuffle=True)
    skf.get_n_splits(df, df[args.label])
    aucs = []
    accs = []
    senss = []
    specs = []
    k = 0



    for train_index, test_index in skf.split(df, df[args.label]):

        k+=1
        trainval_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        trainloader, valloader, testloader, nrois = prepare_kfold_loaders(args, trainval_df, test_df)
        model_params = {'n_conv_layers': args.n_conv_layers, 'n_s4_layers': args.n_s4_layers, 'd_model': args.d_model, 'd_input': nrois, 'T':args.input_len, 'channels':args.channels, 'clf':args.clf, 'lr':args.lr}
        trainer = Trainer(accelerator="gpu", devices=1, max_epochs=args.epochs, check_val_every_n_epoch=1, log_every_n_steps=5,
         callbacks=[ModelCheckpoint(monitor='val_acc', mode='max'), EarlyStopping(monitor='val_acc', patience=15, verbose=True, mode='max')])
        network = mymodel(model_params).to(device)
        cudnn.benchmark = True
        print(f'Training fold : {k}')
        trainer.fit(network, trainloader, valloader)
        trainer.validate(ckpt_path='best', dataloaders=valloader)
        results = trainer.test(ckpt_path='best', dataloaders=testloader)[0]
        print('-'*60)
        aucs.append(results['test_auc'])
        accs.append(results['test_acc'])
        senss.append(results['test_sens'])
        specs.append(results['test_spec'])
        print("for fold {}, Acc = {:.3f}, Sens = {:.3f}, Spec  = {:.3f}".format(k,results['test_acc'],results['test_sens'],results['test_spec']))
        print('-'*30)

    aucs = np.array(aucs)
    auc_mean = np.round(np.mean(aucs),3)
    auc_std = np.round(np.std(aucs),3)
    accs = np.array(accs)
    acc_mean = np.round(np.mean(accs),3)
    acc_std = np.round(np.std(accs),3)
    senss = np.array(senss)
    sens_mean = np.round(np.mean(senss),3)
    sens_std = np.round(np.std(senss),3)
    specs = np.array(specs)
    spec_mean = np.round(np.mean(specs),3)
    spec_std = np.round(np.std(specs),3)

    print ('5-fold results:')
    print(" Test Accuracy: mean = {:.3f} % ,std = {:.3f}".format(acc_mean*100, acc_std*100))
    print(" Test Sens: mean = {:.3f} % ,std = {:.3f}".format(sens_mean*100, sens_std*100))
    print(" Test Spec: mean = {:.3f} % ,std = {:.3f}".format(spec_mean*100, spec_std*100))

    wandb.log({'Test_auc':auc_mean, 'Test_acc':acc_mean, 'Test_sens':sens_mean, 'Test_spec': spec_mean, 'std': [auc_std, acc_std, sens_std, spec_std]})
    shutil.rmtree('./lightning_logs/')
train_kfold(args)
