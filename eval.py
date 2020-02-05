'''
# Reconstruct image and evalute the performance by Generalization
# Author: Yuki Saeki
'''

import os
import argparse
import torch
import torch.utils.data
import cloudpickle
from tqdm import trange, tqdm
import numpy as np
import gudhi as gd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from topologylayer.nn import (PartialSumBarcodeLengths,
                              SumBarcodeLengths, TopKBarcodeLengths)
from utils import *
from topologylayer.nn.levelset import *
from tqdm import trange

parser = argparse.ArgumentParser(description='VAE test')
parser.add_argument('--input', type=str, default="E:/git/pytorch/vae/input/s0/filename.txt",
                    help='File path of input images')
parser.add_argument('--outdir', type=str, default="E:/git/DAE/results/artificial/z_24/B_0.1/R_0/",
                    help='File path of output images')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--constrain', '-c', type=bool, default=False, help='topo con')
parser.add_argument('--mode', type=int, default=0,
                    help='[mode: process] = [0: artificial], [1: real]')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.mode == 0:
    num_of_data = 10000
    num_of_test = 2000
    num_of_val = 2000
elif args.constrain == True:
    num_of_data = 1978
    num_of_test = 467
    num_of_val = 425
else:
    num_of_data = 3039
    num_of_test = 607
    num_of_val = 607

patch_side = 9
latent_dim = 24

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

# check folder
if not (os.path.exists(args.outdir + 'gen')):
    os.makedirs(args.outdir + 'gen/ori/')
    os.makedirs(args.outdir + 'gen/noisy/')
    os.makedirs(args.outdir + 'gen/rec/')
    os.makedirs(args.outdir + 'spe/')

model_path = args.outdir+"model.pkl"

# get data
data_set = get_dataset(args.input, patch_side, num_of_data)
data = data_set.reshape(num_of_data, patch_side * patch_side * patch_side)
data = min_max(data, axis=1)
test = data[:num_of_test]

# divide data
test_data = torch.from_numpy(test).float().to(device)
train_data = torch.from_numpy(data[num_of_test+num_of_val:]).float().to(device)
# test_data = add_noise(test_data, device)
train_data = add_noise(train_data, device)
train_loader = torch.utils.data.DataLoader(train_data,
                          batch_size=train_data.size(0),
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False,
                          drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False,
                          drop_last=False)
# load model
with open(model_path, 'rb') as f:
    model = cloudpickle.load(f)

def gen(model):
    model.eval()
    with torch.no_grad():
        file_ori = open(args.outdir + 'gen/ori/list.txt', 'w')
        file_noisy = open(args.outdir + 'gen/noisy/list.txt', 'w')
        file_rec = open(args.outdir + 'gen/rec/list.txt', 'w')

        bar01_o = []
        bar0_o = []
        bar1_o = []
        bar2_o = []
        bar01_n = []
        bar0_n = []
        bar1_n = []
        bar2_n = []
        bar01 = []
        bar0 = []
        bar1 = []
        bar2 = []

        for i, data in enumerate(tqdm(test_loader)):
            data = data.to(device)
            noisy_data = add_noise(data, device)
            recon_batch, mu, logvar = model(noisy_data)
            # ori_single = data[0, :]
            noisy_single = noisy_data[0, :]
            rec_single = recon_batch[0, :]
            # original = ori_single.cpu().numpy()
            noisy = noisy_single.cpu().numpy()
            reconstruction = rec_single.cpu().numpy()

            ori = np.reshape(test[i], [patch_side, patch_side, patch_side])
            noisy = np.reshape(noisy, [patch_side, patch_side, patch_side])
            rec = np.reshape(reconstruction, [patch_side, patch_side, patch_side])
            generalization = []

            # EUDT
            ori_image = sitk.GetImageFromArray(ori)
            ori_image.SetOrigin([0, 0, 0])
            ori_image.SetSpacing([0.885, 0.885, 1])

            noisy_image = sitk.GetImageFromArray(noisy)
            noisy_image.SetOrigin([0, 0, 0])
            noisy_image.SetSpacing([0.885, 0.885, 1])

            rec_image = sitk.GetImageFromArray(rec)
            rec_image.SetOrigin([0, 0, 0])
            rec_image.SetSpacing([0.885, 0.885, 1])

            # output image
            io.write_mhd_and_raw(ori_image, '{}.mhd'.format(os.path.join(args.outdir, 'gen/ori', '{}'.format(str(i).zfill(4)))))
            io.write_mhd_and_raw(noisy_image, '{}.mhd'.format(os.path.join(args.outdir, 'gen/noisy', '{}'.format(str(i).zfill(4)))))
            io.write_mhd_and_raw(rec_image, '{}.mhd'.format(os.path.join(args.outdir, 'gen/rec', '{}'.format(str(i).zfill(4)))))
            file_ori.write('{}.mhd'.format(os.path.join(args.outdir, 'gen/ori', '{}'.format(str(i).zfill(4)))) + "\n")
            file_noisy.write('{}.mhd'.format(os.path.join(args.outdir, 'gen/noisy', '{}'.format(str(i).zfill(4)))) + "\n")
            file_rec.write('{}.mhd'.format(os.path.join(args.outdir, 'gen/rec', '{}'.format(str(i).zfill(4)))) + "\n")

            # calculate generalization
            generalization.append(L1norm(ori, rec))

            # # calculate PH
            # b01_o, b0_o, b1_o, b2_o = PH(ori)
            # bar01_o.append(b01_o.item())
            # bar0_o.append(b0_o.item())
            # bar1_o.append(b1_o.item())
            # bar2_o.append(b2_o.item())
            #
            # # calculate PH
            # b01_n, b0_n, b1_n, b2_n = PH(noisy)
            # bar01_n.append(b01_n.item())
            # bar0_n.append(b0_n.item())
            # bar1_n.append(b1_n.item())
            # bar2_n.append(b2_n.item())

            # calculate PH
            b01, b0, b1, b2 = PH(rec)
            bar01.append(b01.item())
            bar0.append(b0.item())
            bar1.append(b1.item())
            bar2.append(b2.item())

    file_ori.close()
    file_noisy.close()
    file_rec.close()
    bar_o = [bar01_o, bar0_o, bar1_o, bar2_o]
    bar_o = np.transpose(bar_o)
    bar_n = [bar01_n, bar0_n, bar1_n, bar2_n]
    bar_n = np.transpose(bar_n)
    bar = [bar01, bar0, bar1, bar2]
    bar = np.transpose(bar)

    return generalization, bar_o, bar_n, bar

# testing
def spe(model):
    specificity = []
    bar01 = []
    bar0 = []
    bar1 = []
    bar2 = []

    #  calculate mu and sigma
    with torch.no_grad():
        train_data_cuda = train_data.to(device)
        recon_batch, mean, logvar = model(train_data_cuda)
    mu = mean.mean().item()
    sigma = (torch.exp(0.5 * logvar)).mean().item()

    model.eval()
    file_gen = open(args.outdir + 'spe/list.txt', 'w')
    with torch.no_grad():
        ori = np.reshape(test, [num_of_test, patch_side, patch_side, patch_side])
        for j in trange(2000):
            sample_z = torch.normal(mu, sigma, (1, latent_dim), generator=torch.manual_seed(args.seed)).to(device)
            gen_batch = model.decode(sample_z)
            gen_single = gen_batch.cpu().numpy()
            gen = np.reshape(gen_single, [patch_side, patch_side, patch_side])
            # EUDT
            eudt_image = sitk.GetImageFromArray(gen)
            eudt_image.SetSpacing([0.885, 0.885, 1])
            eudt_image.SetOrigin([0, 0, 0])

            # calculate spe
            case_min_specificity = 1.0
            for image_index in range(num_of_test):
                specificity_tmp = L1norm(ori[image_index] ,gen)
                if specificity_tmp < case_min_specificity:
                    case_min_specificity = specificity_tmp
            specificity.append([case_min_specificity])

            # output image
            io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(args.outdir, 'spe', '{}'.format(str(j).zfill(4)))))
            file_gen.write('{}.mhd'.format(os.path.join(args.outdir, 'spe', '{}'.format(str(j).zfill(4)))) + "\n")
            # calculate PH
            b01, b0, b1, b2 = PH(gen)
            bar01.append(b01.item())
            bar0.append(b0.item())
            bar1.append(b1.item())
            bar2.append(b2.item())

        bar = [bar01, bar0, bar1, bar2]
        bar = np.transpose(bar)
        file_gen.close()

    return specificity, bar

def PH(data):
    z, y, x = data.shape
    cpx = init_tri_complex_3d(z, y, x)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    dgminfo = layer(torch.from_numpy(data).float())
    f01 = TopKBarcodeLengths(dim=0, k=1)
    f0 = PartialSumBarcodeLengths(dim=0, skip=1)
    f1 = SumBarcodeLengths(dim=1)
    f2 = SumBarcodeLengths(dim=2)
    b01 = f01(dgminfo).sum()
    b0 = f0(dgminfo)
    b1 = f1(dgminfo)
    b2 = f2(dgminfo)
    return b01, b0, b1, b2

if __name__ == "__main__":

    # generalization
    generalization, bar_o, bar_n, bar = gen(model)
    print('generalization = %f' % np.mean(generalization))
    np.savetxt(os.path.join(args.outdir, 'generalization.csv'), generalization, delimiter=",")
    # np.savetxt(os.path.join(args.outdir, 'ori_topo.csv'), bar_o, delimiter=",")
    # np.savetxt(os.path.join(args.outdir, 'noisy_topo.csv'), bar_n, delimiter=",")
    np.savetxt(os.path.join(args.outdir, 'gen_topo.csv'), bar, delimiter=",")

    # specificity
    specificity, bar = spe(model)
    print('specificity = %f' % np.mean(specificity))
    np.savetxt(os.path.join(args.outdir, 'specificity.csv'), specificity, delimiter=",")
    np.savetxt(os.path.join(args.outdir, 'spe_topo.csv'), bar, delimiter=",")