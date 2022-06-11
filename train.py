import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from model import X_vector
from generator import SpeechDataset
import generator
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from adacos import AdaCos 

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def train(model, loader, optimizer, scheduler,criterion,
          epoch, iter_meter):
    model.train()

    data_len=len(loader)

    epoch_loss = 0
    preds=None
    corrs=None
    for batch_idx, _data in enumerate(loader):
        inputs, speakers, lengths = _data

        optimizer.zero_grad()

        outputs, logits = model(inputs.cuda(), speakers.cuda(), lengths)
        loss = criterion(outputs, speakers.cuda())
        #loss, outputs = criterion(x_vector, speakers.cuda())
        loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()

        speakers=speakers.unsqueeze(dim=-1)
        logits=logits.to('cpu').detach().numpy().copy()
        speakers=speakers.to('cpu').detach().numpy().copy().astype(float)
        if preds is None:
            preds=logits
            corrs=speakers
        else:
            preds = np.vstack((preds, logits))
            corrs = np.vstack((corrs, speakers))

        if batch_idx > 0 and (batch_idx % 100 == 0 or batch_idx == data_len) :
            print('Train Epcoh: {} [{}/{} ({:.0f}%)]\t Loss: {:.9f}'.format(
                epoch, batch_idx * len(inputs), data_len*inputs.shape[0],
                100. * batch_idx / data_len, loss.item())
            )

        epoch_loss += loss.item()

        del loss
        torch.cuda.empty_cache()

    epoch_loss /= data_len

    print('Train Epcoh: {}\t Loss: {:.9f}'.format(epoch, epoch_loss))
    preds = np.argmax(preds, axis=1)
    return epoch_loss, preds, corrs

def evaluate(model, loader):

    model.eval()
    data_len = len(loader)
    preds=None
    corrs=None

    with torch.no_grad():
        for i, _data in enumerate(loader):
            inputs, speakers, lengths = _data

            outputs, logits = model(inputs.cuda(), speakers=None, lengths=lengths)
            #outputs = criterion(x_vector)
            speakers = speakers.unsqueeze(dim=-1)

            logits=logits.to('cpu').detach().numpy().copy()
            speakers=speakers.to('cpu').detach().numpy().copy().astype(float)
            if preds is None:
                preds=logits
                corrs=speakers
            else:
                preds = np.vstack((preds, logits))
                corrs = np.vstack((corrs, speakers))
                
    preds = np.argmax(preds, axis=1)
    return preds, corrs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data(.h5)')
    parser.add_argument('--train', type=str, required=True,
                        help='training keys')
    parser.add_argument('--valid', type=str, required=True,
                        help='validation keys')
    parser.add_argument('--out-stats', type=str, default='stats.h5',
                        help='mean/std values for norm')
    parser.add_argument('--out-speakers', type=str, default='speakers.txt',
                        help='speakers id')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='training epochs')
    parser.add_argument('--output', type=str, default='model',
                        help='output model')
    parser.add_argument('--learning-rate', type=float, default=1.0e-4,
                        help='initial learning rate')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda is True:
        print('use GPU')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset=SpeechDataset(path=args.data, keypath=args.train, train=True)
    mean, std = train_dataset.get_stats()
    s2i, i2s = train_dataset.get_speakers()
    train_dataset.write_stats(args.out_stats)
    train_dataset.write_speakers(args.out_speakers)
    num_speakers=len(s2i)
    
    train_loader =data.DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=lambda x: generator.data_processing(x,'train'),
                                  **kwargs)

    valid_dataset=SpeechDataset(path=args.data, keypath=args.valid,
                                stats=[mean, std], speakers=[s2i, i2s], train=False)
    valid_loader=data.DataLoader(dataset=valid_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda x: generator.data_processing(x, 'valid'))

    model=X_vector(train_dataset.input_size(), num_speakers)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr = args.learning_rate,
                                betas=(0.9, 0.98),
                                eps=1.0e-9)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=args.learning_rate,
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=args.epochs,
                                              anneal_strategy='linear')
    criterion=nn.CrossEntropyLoss().cuda()
    iter_meter=IterMeter()

    max_acc=0.
    for epoch in range(1, args.epochs+1):
        loss, preds, corrs = train(model, train_loader,
                                   optimizer, scheduler,criterion,
                                   epoch, iter_meter)
        acc = accuracy_score(corrs, preds)
        print('Train Acc: %.3f' % acc)
        # get predictions shape=(B, 1)
        preds, corrs = evaluate(model, valid_loader)
        acc = accuracy_score(corrs, preds)

        print('Valid Acc: %.3f' % acc)

        if acc > max_acc:
            print('Maximum Acc changed... %.3f -> %.3f' % (max_acc , acc))
            max_acc = acc
            #torch.save(model.to('cpu').state_dict(), args.output)
            torch.save(model.to('cpu'), args.output)
            model.to(device)
            
    print('Maximum Acc: %.3f' % max_acc)

if __name__ == "__main__":
    main()
