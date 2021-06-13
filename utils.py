import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py

def read_speakers(file):
    id2spk={}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            spk = line.strip().split()[0]
            id = int(line.strip().split()[1])
            self.id2spk[id] = spk
    return id2spk

def read_stats(file):
    with h5py.File(file, 'r') as f:
        mean=f['mean'][()]
        std=f['std'][()]

    return mean, std

