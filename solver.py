from topologylayer.nn.features import TopKBarcodeLengths, get_barcode_lengths
from topologylayer.nn.levelset import LevelSetLayer
from topologylayer.util.construction import unique_simplices
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(9 ** 3, 200)
        self.fc21 = nn.Linear(200, latent_dim)
        self.fc22 = nn.Linear(200, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, 9 ** 3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 9 ** 3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class SquaredBarcodeLengths(nn.Module):
    """
    Layer that sums up lengths of barcode in persistence diagram
    ignores infinite bars, and padding
    Options:
        dim - barcode dimension to sum over (default 0)

    forward input:
        (dgms, issub) tuple, passed from diagram layer
    """

    def __init__(self, dim=0):
        super(SquaredBarcodeLengths, self).__init__()
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)

        # return Squared norm of the barcode lengths
        return torch.sum(lengths ** 2, dim=0)


class PartialSquaredBarcodeLengths(nn.Module):
    """
    Layer that computes a partial Squared lengths of barcode lengths

    inputs:
        dim - homology dimension
        skip - skip this number of the longest bars

    ignores infinite bars and padding
    """

    def __init__(self, dim, skip):
        super(PartialSquaredBarcodeLengths, self).__init__()
        self.skip = skip
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)

        # sort lengths
        sortl, indl = torch.sort(lengths, descending=True)

        return torch.sum(sortl[self.skip:] ** 2)


def init_tri_complex_3d(width, height, depth):
    """
    initialize 3d complex in dumbest possible way
    """
    # initialize complex to use for persistence calculations
    axis_x = np.arange(0, width)
    axis_y = np.arange(0, height)
    axis_z = np.arange(0, depth)
    grid_axes = np.array(np.meshgrid(axis_x, axis_y, axis_z))
    grid_axes = np.transpose(grid_axes, (1, 2, 3, 0))

    # creation of a complex for calculations
    tri = Delaunay(grid_axes.reshape([-1, 3]))
    return unique_simplices(tri.simplices, 3)


class TopLoss(nn.Module):
    def __init__(self):
        super(TopLoss, self).__init__()
        self.size = 9
        self.cpx = init_tri_complex_3d(self.size, self.size, self.size)
        self.pdfn = LevelSetLayer(self.cpx, maxdim=2, sublevel=False)
        self.topfn0 = TopKBarcodeLengths(dim=0, k=1)
        self.fn0 = PartialSquaredBarcodeLengths(dim=0, skip=1)  # penalize more than 1 cc
        self.fn1 = SquaredBarcodeLengths(dim=1)
        self.fn2 = SquaredBarcodeLengths(dim=2)

    def forward(self, data):
        dgminfo = self.pdfn(data)
        t01 = (1. - self.topfn0(dgminfo) ** 2).sum()
        t0 = self.fn0(dgminfo).sum()
        t1 = self.fn1(dgminfo).sum()
        t2 = self.fn2(dgminfo).sum()
        loss = t01 + t0 + t1 + t2
        return loss, t01, t0, t1, t2


def shift(img, val, dim, device='cuda'):
    img = torch.roll(img, val, dim)
    side = img.size()[dim] - 1
    if val >= 0:
        idx = torch.arange(val, dtype=int)
        bd_idx = torch.zeros(val, dtype=int) + val
    else:
        idx = side - torch.arange(-val, dtype=int)
        bd_idx = side - torch.zeros(-val, dtype=int) + val
    idx, bd_idx = idx.to(device=device), bd_idx.to(device=device)
    return img.index_copy_(dim, idx, torch.index_select(img, dim, bd_idx))


def erosion_layer(img):
    # img.size() = (N, C, W, H, D)
    nbh = torch.cat((img, shift(img, -1, 2), shift(img, 1, 2),
                     shift(img, -1, 3), shift(img, 1, 3),
                     shift(img, -1, 4), shift(img, 1, 4)), 1)
    erosion, _ = torch.min(nbh, 1)
    return erosion
