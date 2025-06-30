import argparse
import os
import random


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    # For faster
    torch.set_float32_matmul_precision('high')   
from tqdm.auto import tqdm

from data.custom_datasets import SortDataset
from models.ctm_sort import ContinuousThoughtMachineSORT
from tasks.image_classification.plotting import plot_neural_dynamics, make_classification_gif
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import sort_loss
from tasks.sort.utils import compute_ctc_accuracy, decode_predictions
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

import torchvision
torchvision.disable_beta_transforms_warning()

from autoclip.torch import QuantileClip

n_to_sort = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adaptive_eviction = True

model = ContinuousThoughtMachineSORT(
    iterations=50,
    d_model=512,
    d_input=n_to_sort,  
    heads=4,
    n_synch_out=32,
    n_synch_action=32,
    synapse_depth=4,
    memory_length=25,  
    deep_nlms=True,
    memory_hidden_dims=4,  
    do_layernorm_nlm=False,  
    backbone_type='none',
    positional_embedding_type='none',
    out_dims=n_to_sort+1,
    prediction_reshaper=[-1],
    dropout=0.0,      
    dropout_nlm=None,    
    neuron_select_type='random-pairing',
    n_random_pairing_self=0,
    adaptive_eviction=adaptive_eviction,
).to(device)

batch_size = 1
batch_size_test = 1
# Data
train_data = SortDataset(n_to_sort)
test_data = SortDataset(n_to_sort)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True, drop_last=False)


print(f'Reloading from: logs/sort/ctm_adaptive_eviction_30/checkpoint.pt')
checkpoint = torch.load(f'logs/sort/ctm_adaptive_eviction_30/checkpoint.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
# print(f'Total params: {sum(p.numel() for p in model.parameters())}')
print("model loaded")

model.eval()
with torch.inference_mode():

    inputs, targets = next(iter(testloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    predictions, certainties, synchronisation, pre_activations, post_activations, _ , evictions = model(inputs, track=True)

    decoded = [d[:targets.shape[1]] for d in decode_predictions(predictions, predictions.shape[1]-1)]
    decoded = torch.stack([torch.concatenate((d, torch.zeros(targets.shape[1] - len(d), device=targets.device)+targets.shape[1])) if len(d) < targets.shape[1] else d for d in decoded], 0)

print("evictions", evictions.shape)
print(evictions)
print("decoded predictions", decoded.shape)
print(decoded)


