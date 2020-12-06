'''
# Training model
# Author: Yuki Saeki
# Reference: https://github.com/JamesClough/topograd, https://github.com/bruel-gabrielsson/TopologyLayer, https://github.com/pytorch/examples/tree/master/vae
'''

from __future__ import print_function
import argparse
import os
import json
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
import cloudpickle
import numpy as np
from tqdm import trange
from topologylayer.nn import LevelSetLayer, TopKBarcodeLengths, SumBarcodeLengths, PartialSumBarcodeLengths
import dataIO as io
from solver import VAE, init_tri_complex_3d, TopLoss, erosion_layer
from utils import add_noise, min_max

parser = argparse.ArgumentParser(description='Topological-DVAE')
parser.add_argument('--input', type=str, default="E:/git/pytorch/vae/input/hole/rank/filename.txt",
                    help='File path of input images')
parser.add_argument('--output', type=str, default="./results/",
                    help='File path of input images')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--beta', type=float, default=1, metavar='B',
                    help='beta')
parser.add_argument('--lam', type=float, default=0, metavar='L',
                    help='lambda')
parser.add_argument('--topo', '-t', action='store_true', default=True, help='topo')
parser.add_argument('--erosion', '-e', action='store_true', default=True, help='use erosion layer')
parser.add_argument('--mode', type=int, default=0,
                    help='[mode: process] = [0: Train], [1: Debug]')
parser.add_argument('--model', type=str, default="",
                    help='File path of loaded model')
parser.add_argument('--latent_dim', type=int, default=24,
                    help='dimension of latent space')
parser.add_argument('--patient', type=int, default=10,
                    help='epochs for early stopping')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

viz = Visdom()

# setting
patch_side = 9
num_of_data = 10000
num_of_test = 2000
num_of_val = 2000
data_path = args.input
outdir = args.output

if not (os.path.exists(outdir)):
    os.makedirs(outdir)

# save parameters
with open(os.path.join(outdir, "params.json"), mode="w") as f:
    json.dump(args.__dict__, f, indent=4)

writer = SummaryWriter(log_dir=outdir + "logs")

print('-' * 20, 'loading data', '-' * 20)
list = io.load_list(data_path)
data_set = np.zeros((len(list), patch_side, patch_side, patch_side))

for i in trange(len(list)):
    data_set[i, :] = np.reshape(io.read_mhd_and_raw(list[i]), [patch_side, patch_side, patch_side])

data = data_set.reshape(num_of_data, patch_side * patch_side * patch_side)
data = min_max(data, axis=1)

# split data
test_data = torch.from_numpy(data[:num_of_test]).float()
val_data = torch.from_numpy(data[num_of_test:num_of_test + num_of_val]).float().to(device)
train_data = torch.from_numpy(data[num_of_test + num_of_val:]).float().to(device)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=False,
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=0,
                                         pin_memory=False,
                                         drop_last=True)

# initialize list for plot graph after training
train_loss_list, val_loss_list = [], []

# define model
if args.model:
    with open(args.model, 'rb') as f:
        model = cloudpickle.load(f).to(device)
    summary(model, (1, patch_side ** 3))
else:
    model = VAE(args.latent_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    batch_size = x.size(0)
    feature_size = x.size(1)
    assert batch_size != 0

    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, patch_side**3), reduction='sum')
    REC = F.mse_loss(recon_x, x, size_average=False, reduction='sum').div(batch_size)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).div(batch_size)
    KLD *= args.beta

    if args.topo == True:
        topo, b01, b0, b1, b2 = 0., 0., 0., 0., 0.
        topo_e, b01_e, b0_e, b1_e, b2_e = 0., 0., 0., 0., 0.
        if args.erosion == True:
            erosion_img = erosion_layer(recon_x.view(batch_size, 1, patch_side, patch_side, patch_side))
        for i in range(batch_size):
            data = recon_x.view(batch_size, patch_side, patch_side, patch_side)[i]
            topo_loss = TopLoss()
            t, t01, t0, t1, t2 = topo_loss(data)
            topo += t
            b01 += t01
            b0 += t0
            b1 += t1
            b2 += t2
            if args.erosion == True:
                e_data = erosion_img.view(batch_size, patch_side, patch_side, patch_side)[i]
                t_e, t01_e, t0_e, t1_e, t2_e = topo_loss(e_data)
                topo_e += t_e
                b01_e += t01_e
                b0_e += t0_e
                b1_e += t1_e
                b2_e += t2_e
        topo, b01, b0, b1, b2 = topo * args.lam / batch_size, b01 * args.lam / batch_size, b0 * args.lam / batch_size, b1 * args.lam / batch_size, b2 * args.lam / batch_size
        total_loss = REC + KLD + topo
        if args.erosion == True:
            topo_e, b01_e, b0_e, b1_e, b2_e = topo_e * args.lam / batch_size, b01_e * args.lam / batch_size, b0_e * args.lam / batch_size, b1_e * args.lam / batch_size, b2_e * args.lam / batch_size
            total_loss += topo_e

        return total_loss, REC, KLD, topo, b01, b0, b1, b2

    else:
        total_loss = REC + KLD
        return total_loss, REC, KLD


def topological_loss(recon_x):
    batch_size = recon_x.size(0)
    b01, b0, b1, b2 = 0., 0., 0., 0.
    cpx = init_tri_complex_3d(patch_side, patch_side, patch_side)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    f01 = TopKBarcodeLengths(dim=0, k=1)
    f0 = PartialSumBarcodeLengths(dim=0, skip=1)
    f1 = SumBarcodeLengths(dim=1)
    f2 = SumBarcodeLengths(dim=2)
    for i in range(batch_size):
        dgminfo = layer(recon_x.view(batch_size, patch_side, patch_side, patch_side)[i])
        b01 += ((1. - f01(dgminfo))).sum()
        b0 += (f0(dgminfo)).sum()
        b1 += (f1(dgminfo)).sum()
        b2 += (f2(dgminfo)).sum()
    b01 = b01.div(batch_size)
    b0 = b0.div(batch_size)
    b1 = b1.div(batch_size)
    b2 = b2.div(batch_size)
    topo = b01 + b0 + b1 + b2
    return topo, b01, b0, b1, b2


def train(epoch):
    model.train()
    train_loss = 0.
    SE, KLD = 0., 0.
    topo = 0.
    b01, b0, b1, b2 = 0., 0., 0., 0.
    for batch_idx, data in enumerate(train_loader):
        noisy_data = add_noise(data, device)
        data = data.to(device)
        noisy_data = noisy_data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(noisy_data)
        if args.mode == 1:
            loss, l01, l0, l1, l2 = topological_loss(recon_batch)
            train_loss += loss.item()
            b01 += l01.item()
            b0 += l0.item()
            b1 += l1.item()
            b2 += l2.item()
        elif args.topo == True:
            loss, l_SE, l_KLD, l_topo, l01, l0, l1, l2 = loss_function(recon_batch, data, mu, logvar)
            train_loss += loss.item()
            SE += l_SE.item()
            KLD += l_KLD.item()
            topo += l_topo.item()
            b01 += l01.item()
            b0 += l0.item()
            b1 += l1.item()
            b2 += l2.item()
        else:
            loss, l_SE, l_KLD = loss_function(recon_batch, data, mu, logvar)
            train_loss += loss.item()
            SE += l_SE.item()
            KLD += l_KLD.item()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(noisy_data)))

    train_loss /= len(train_loader)

    train_loss_list.append(train_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))
    if args.mode == 0:
        SE /= len(train_loader)
        KLD /= len(train_loader)
        writer.add_scalars("loss/each_loss", {'Train': train_loss,
                                              'Rec': SE,
                                              'KL': KLD,
                                              'Topo': topo}, epoch)
        writer.add_scalars("loss/each_loss", {'Train': train_loss,
                                              'Rec': SE,
                                              'KL': KLD}, epoch)

        if args.topo == True:
            b01 /= len(train_loader)
            b0 /= len(train_loader)
            b1 /= len(train_loader)
            b2 /= len(train_loader)
            topo /= len(train_loader)
            writer.add_scalars("loss/topological_loss", {'topo': topo,
                                                         'b01': b01,
                                                         'b0': b0,
                                                         'b1': b1,
                                                         'b2': b2}, epoch)

    return train_loss


def val(epoch):
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            noisy_val_data = add_noise(val_data, device)
            noisy_val_data = noisy_val_data.to(device)
            val_data = val_data.to(device)
            recon_batch, mu, logvar = model(noisy_val_data)
            if args.mode == 1:
                loss, _, _, _, _ = topological_loss(recon_batch)
            elif args.topo == True:
                loss, l_SE, l_KLD, l_topo, l01, l0, l1, l2 = loss_function(recon_batch, val_data, mu, logvar)
            else:
                loss, l_SE, l_KLD = loss_function(recon_batch, val_data, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_loss_list.append(val_loss)
    viz.line(X=np.array([epoch]), Y=np.array([val_loss]), win='val_loss', name='validation', update='append',
             opts=dict(showlegend=True))
    print('====> val set loss: {:.4f}'.format(val_loss))

    return val_loss


if __name__ == "__main__":
    val_loss_min = np.inf
    min_delta = 0.001
    epochs_no_improve = 0
    n_epochs_stop = args.patient

    for epoch in trange(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = val(epoch)
        writer.add_scalars("loss/total_loss", {'train': train_loss,
                                               'val': val_loss}, epoch)
        viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='loss', name='train_loss', update='append',
                 opts=dict(showlegend=True))
        viz.line(X=np.array([epoch]), Y=np.array([val_loss]), win='loss', name='val_loss', update='append')
        with open(outdir + 'train_loss', 'wb') as f:
            cloudpickle.dump(train_loss_list, f)
        with open(outdir + 'val_loss', 'wb') as f:
            cloudpickle.dump(val_loss_list, f)

        if val_loss < val_loss_min - min_delta:
            epochs_no_improve = 0
            val_loss_min = val_loss
            # save model
            if epoch > args.log_interval:
                path = os.path.join(outdir, 'weight/')
                if not (os.path.exists(path)):
                    os.makedirs(path)
                torch.save(model.state_dict(), path + '{}epoch-{}.pth'.format(epoch, round(val_loss, 2)))
            with open(outdir + 'model.pkl', 'wb') as f:
                cloudpickle.dump(model, f)
        else:
            epochs_no_improve += 1

        # Check early stopping condition
        if epochs_no_improve >= n_epochs_stop:
            print('-' * 20, 'Early stopping!', '-' * 20)
            break
