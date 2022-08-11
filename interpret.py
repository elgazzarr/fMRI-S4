import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model import mymodel
import numpy as np
import argparse
from datasets import Ukbb, Mddrest, Abide, get_dataset_class
import pickle as pkl
import itertools
import pandas as pd
from tqdm import tqdm
from attribution.mask import Mask
from attribution.perturbation import FadeMovingAverage
from utils.losses import *
from attribution.mask_group import Mask
from attribution.perturbation import GaussianBlur
from captum.attr import (
    FeaturePermutation,
    GradientShap,
    IntegratedGradients,
    Occlusion,
    ShapleyValueSampling,
    NoiseTunnel
)
parser = argparse.ArgumentParser(description='Training ')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

parser.add_argument('--label', default='Diagnosis',  choices=['Sex', 'MDD', 'Diagnosis'], type=str, help= 'label to detect')
parser.add_argument('--dataset', default='Synth', choices=['Ukbb', 'Mddrest', 'Abide', 'Jpmdd', 'Synth'], type=str, help='Dataset')
parser.add_argument('--atlas', default='HO_112', type=str, help='Parcellation atlas ')
parser.add_argument('--input_len', default='250', type=int, help='length of the timeseries')
# Dataloader
parser.add_argument('--num_workers', default=5, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
# Model
parser.add_argument('--n_conv_layers', default=1, type=int, help='Number of conv layers in encoder')
parser.add_argument('--n_s4_layers', default=2, type=int, help='Number of S4 layers')
parser.add_argument('--d_model', default=512, type=int, help='Model dimension')
parser.add_argument('--clf', default='B', type=str, help='Classifier head option')
parser.add_argument('--channels', default=3, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0, type=float, help='Dropout')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_mask():
    networks = np.load('../state-spaces/ho_networks.npy', allow_pickle=True).item()
    n = list(networks.values())
    rois_indicies = list(itertools.chain(*n))
    mask = np.zeros(118)
    for i, network in enumerate(networks):
        rois_ind = networks[network]
        mask[rois_ind] = i+1
    return torch.tensor(mask)



def interpret(model_path, args):

    testset =  get_dataset_class(args.dataset)(pd.read_csv('./csvfiles/{}_test.csv'.format(args.dataset)), args.atlas, args.input_len)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=10)

    model_params = {'n_conv_layers': args.n_conv_layers, 'n_s4_layers': args.n_s4_layers, 'd_model': args.d_model, 'd_input': 160, 'T':args.input_len, 'channels':args.channels, 'clf':args.clf, 'lr':args.lr}

    network = mymodel(model_params).load_from_checkpoint(model_path).to(device)
    #network.eval()
    X,Y = next(iter(testloader))
    X = X.transpose(2,1).to(device)

    mask_ig = torch.zeros(size=X.shape, device=device)
    pert = FadeMovingAverage(device)
    dynamask = torch.zeros(size=X.shape, dtype=torch.float32, device=device)

    def f(x):
        x = x.unsqueeze(0)
        out = network(x)
        out = torch.nn.Softmax(dim=-1)(out)
        out = out[0]
        return out

    #feature premuatation
    #feature_perm = FeaturePermutation(network)
    #feature_mask = torch.tile(torch.arange(0,testset.nrois),(testset.ntime,1)).transpose(1,0).to(device)
    #atr_fp = feature_perm.attribute(X, target=1, feature_mask=feature_mask.unsqueeze(0), show_progress=True)

    #Integrated Gradients
    integrated_gradients = IntegratedGradients(network)
    for i,x in tqdm(enumerate(X[0:3])):
        #x = x.unsqueeze(0)
        #atr_ig = integrated_gradients.attribute(x, target=torch.argmax(network(x)), n_steps=200)
        #mask_ig[i] = atr_ig[0]

        mask = Mask(pert, device, task="classification", verbose=False, deletion_mode=True)
        mask.fit(X=x,
                f=f,
                loss_function=cross_entropy,
                keep_ratio=0.01,
                n_epoch=1000,
                learning_rate=2)
        dynamask[i] = mask.mask_tensor

    #with open(f"./visulizations/Synth_mask_networks_2_ig.pkl", "wb") as file:
        #pkl.dump(mask_ig.detach().cpu().numpy(), file)

    with open(f"./visulizations/Synth_mask_networks_2_dynmask.pkl", "wb") as file:
        pkl.dump(dynamask.detach().cpu().numpy(), file)

    #with open(f"./visulizations/Synth_mask_networks_2.pkl", "wb") as file:
        #pkl.dump(atr.detach().cpu().numpy(), file)

model_path = "./ckpts/lightning_logs/version_10/checkpoints/epoch=12-step=51.ckpt"
interpret(model_path, args)
