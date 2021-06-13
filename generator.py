import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import torch
import torch.nn as nn

def compute_spk2id(h5fd):
    spk2id={}
    id2spk={}
    spk2id['<unk>']=0
    id2spk[0]='<unk>'

    n=1
    for key in h5fd.keys():
        spk=h5fd[key+'/speaker'][()].decode('utf-8')
        if spk not in spk2id:
            spk2id[spk] = n
            id2spk[n]=spk
            n+=1
    return spk2id, id2spk

def compute_norm(h5fd):
    keys=h5fd.keys()
    rows=0
    mean=None
    std=None

    for key in keys:
        if key == 'mean':
            continue
        if key == 'std':
            continue
        mat = h5fd[key+'/data'][()]
        rows += mat.shape[0]
        if mean is None:
            mean=np.sum(mat, axis=0).astype(np.float64)
            std=np.sum(np.square(mat), axis=0).astype(np.float64)
        else:
            mean=np.add(np.sum(mat, axis=0).astype(np.float64), mean)
            std=np.add(np.sum(np.square(mat), axis=0).astype(np.float64), std)

    mean = mean/rows
    std = np.sqrt(std/rows - np.square(mean))

    return mean, std

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, path, keypath=None, stats=None,
                 speakers=None, train=True):
        super(SpeechDataset, self).__init__()

        self.train=train

        self.h5fd = h5py.File(path, 'r')
        if stats is None:
            self.mean, self.std = compute_norm(self.h5fd)
        else:
            self.mean, self.std = stats
        if speakers is None:
            self.spk2id, self.id2spk = compute_spk2id(self.h5fd)
        else:
            self.spk2id, self.id2spk = speakers
            
        self.keys=[]

        if keypath is not None:
            with open(keypath,'r') as f:
                lines=f.readlines()
                for l in lines:
                    self.keys.append(l.strip())

    def write_stats(self, file):
        with h5py.File(file, 'w') as f:
            f.create_dataset('mean',  data=self.mean,
                             compression='gzip', compression_opts=9)
            f.create_dataset('std',  data=self.std,
                             compression='gzip', compression_opts=9)
            
    def get_stats(self):
        return self.mean, self.std

    def get_speakers(self):
        return self.spk2id, self.id2spk

    def write_speakers(self, file):
        with open(file, 'w') as f:
            for spk in self.spk2id:
                f.write("%s %d\n" % (spk, self.spk2id[spk]))
                
    def get_keys(self):
        return self.keys

    def __len__(self):
        return len(self.keys)

    def input_size(self):
        mat = self.h5fd[self.keys[0]+'/data'][()]
        return mat.shape[1]

    def get_data(self, keys):
        data=[]
        for key in keys:
            dt = self.__getitem__(self.keys.index(key))
            data.append(dt)
        _data=data_processing(data)
        return _data

    def __getitem__(self, idx):
        # (time, feature)
        input=self.h5fd[self.keys[idx]+'/data'][()]

        input -= self.mean
        input /= self.std

        speaker=self.spk2id[self.h5fd[self.keys[idx]+'/speaker'][()].decode('utf-8')]
        
        return input, speaker

'''
    data_processing
    Return inputs, labels, input_lengths, label_lengths, outputs
'''
def data_processing(data, data_type="train"):
    inputs = []
    speakers = []
    input_lengths=[]

    for input, speaker, in data:
        """ inputs : (batch, time, feature) """
        # w/o channel
        inputs.append(torch.from_numpy(input.astype(np.float32)).clone())
        speakers.append(speaker)
        
        input_lengths.append(input.shape[0])

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    speakers = torch.from_numpy(np.array(speakers)).clone()
    
    return inputs, speakers, input_lengths
