import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gudhi as gd
import dataIO as io
import torch
from mpl_toolkits.mplot3d import axes3d
from topologylayer.functional.persistence import SimplicialComplex
from topologylayer.util.construction import unique_simplices
from scipy.spatial import Delaunay
from tqdm import trange

# calculate jaccard
def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.bitwise_and(im1, im2).sum()) / np.double(np.bitwise_or(im1, im2).sum())

# calculate L1
def L1norm(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean(abs(im1 - im2)))

# calculate L2
def L2norm(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean((im1 - im2)^2))

def matplotlib_plt(X, filename):
    fig = plt.figure()
    plt.title('latent distribution')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('dim_1')
    ax.set_ylabel('dim_2')
    ax.set_zlabel('dim_3')
    ax.scatter(X[:,0], X[:,1], X[:,2] , marker="x"
               # , c=y/len(set(y))
    )
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.savefig(filename + "3D/{:03d}.jpg".format(angle))
    # plt.savefig(filename)
    # plt.show()

def visualize_slices(X, Xe, outdir):
    # plot reconstruction
    fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(18, 4))
    for i in range(10):
        minX = np.min(X[i, :])
        maxX = np.max(X[i, :])
        axes[0, i].imshow(X[i, :].reshape(9, 9), cmap=cm.Greys_r, vmin=0, vmax=1,
                          interpolation='none')
        axes[0, i].set_title('original %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        minXe = np.min(Xe[i, :])
        maxXe = np.max(Xe[i, :])
        axes[1, i].imshow(Xe[i, :].reshape(9, 9), cmap=cm.Greys_r, vmin=0, vmax=1,
                          interpolation='none')
        axes[1, i].set_title('reconstruction %d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "reconstruction.png")
    # plt.show()

def display_slices(case):
    # case: image data (num_data, size, size, size)
    min = 0
    max = 1
    num_data, z, y, x = case.shape
    if num_data==1:
        case = case.reshape(z, y, x)
        # sagital
        fig, axes = plt.subplots(ncols=x, nrows=num_data, figsize=(x - 3, num_data), dpi=150)
        for i in range(x):
            axes[i].imshow(case[:, :, i].reshape(x, x), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('x = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
        # coronal
        fig, axes = plt.subplots(ncols=y, nrows=num_data, figsize=(y - 3, num_data), dpi=150)
        for i in range(y):
            axes[i].imshow(case[:, i, :].reshape(y, y), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('y = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
        # axial
        fig, axes = plt.subplots(ncols=z, nrows=num_data, figsize=(z - 3, num_data), dpi=150)
        for i in range(z):
            axes[i].imshow(case[i, :, :].reshape(z, z), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('z = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
    else:
        # sagital
        fig, axes = plt.subplots(ncols=x, nrows=num_data, figsize=(x - 2, num_data), dpi=150)
        for i in range(x):
            for j in range(num_data):
                axes[j, i].imshow(case[j, :, :, i].reshape(x, x), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
                axes[j, i].set_title('x = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()
        # coronal
        fig, axes = plt.subplots(ncols=y, nrows=num_data, figsize=(y - 2, num_data), dpi=150)
        for i in range(y):
            for j in range(num_data):
                axes[j, i].imshow(case[j, :, i, :].reshape(y, y), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
                axes[j, i].set_title('y = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()
        # axial
        fig, axes = plt.subplots(ncols=z, nrows=num_data, figsize=(z - 2, num_data), dpi=150)
        for i in range(z):
            for j in range(num_data):
                axes[j, i].imshow(case[j, i, :, :].reshape(z, z), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
                axes[j, i].set_title('z = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()

def display_center_slices(case, size, num_data, outdir):
    # case: image data, num_data: number of data, size: length of a side
    min = np.min(case)
    max = np.max(case)
    # axial
    fig, axes = plt.subplots(ncols=num_data, nrows=1, figsize=(num_data, 2))
    for i in range(num_data):
        axes[i].imshow(case[i, 3, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
        axes[i].set_title('image%d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    # plt.savefig(outdir + "/interpolation.png")
    plt.show()

def init_freudenthal_3d(width, height, depth):
    """
    Freudenthal triangulation of 2d grid
    """
    s = SimplicialComplex()
    # row-major format
    # 0-cells
    for i in range(depth):
        for j in range(height):
            for k in range(width):
               ind = i*width*height + j*width + depth
               s.append([ind])
    # 1-cells
    for i in range(depth):
        for j in range(height):
            for k in range(width-1):
                ind = i * width * height + j * width + depth
                s.append([ind, ind + 1])
    for i in range(depth-1):
        for j in range(width):
            for k in range(width):
                ind = i*width + j
                s.append([ind, ind + width])
    # 2-cells + diagonal 1-cells
    for i in range(depth-1):
        for j in range(width-1):
            for k in range(width):
                ind = i*width + j
                # diagonal
                s.append([ind, ind + width + 1])
                # 2-cells
                s.append([ind, ind + 1, ind + width + 1])
                s.append([ind, ind + width, ind + width + 1])
    return s

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

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

def diag_tidy(diag, eps=1e-1):
    new_diag = []
    for _, x in diag:
        if np.abs(x[0] - x[1]) > eps:
            new_diag.append((_, x))
    return new_diag

def PH_diag(img):
    z, y, x = img.shape
    cc = gd.CubicalComplex(dimensions=(z, y, x),
                           top_dimensional_cells=1 - img.flatten())
    diag = cc.persistence()
    plt.figure(figsize=(3, 3))
    diag_clean = diag_tidy(diag, 1e-3)
    gd.plot_persistence_barcode(diag, max_intervals=0,inf_delta=100)
    print(diag)
    plt.xlim(0, 1)
    plt.ylim(-1, len(diag))
    plt.xticks(ticks=np.linspace(0, 1, 6), labels=np.round(np.linspace(1, 0, 6), 2))
    plt.yticks([])
    plt.show()
    gd.plot_persistence_diagram(diag, legend=True)
    plt.show()
    gd.plot_persistence_density(diag, legend=True)
    plt.show()

def save_PH_diag(img, patch_side, outdir):
    cc = gd.CubicalComplex(dimensions=(patch_side, patch_side, patch_side),
                           top_dimensional_cells=1 - img.flatten())
    diag = cc.persistence()
    plt.figure(figsize=(3, 3))
    diag_clean = diag_tidy(diag, 1e-3)
    print(diag_clean)
    # np.savetxt(os.path.join(outdir, 'generalization.csv'), diag_clean, delimiter=",")
    with open(os.path.join(outdir, 'generalization.txt'), 'wt') as f:
        for ele in diag_clean:
            f.write(ele + '\n')
    gd.plot_persistence_barcode(diag_clean)
    plt.ylim(-1, len(diag_clean))
    plt.xticks(ticks=np.linspace(0, 1, 6), labels=np.round(np.linspace(1, 0, 6), 2))
    plt.yticks([])
    plt.savefig(os.path.join(outdir, "PH_diag.png"))

def get_dataset(input, patch_side, num_of_test):
    print('load data')
    list = io.load_list(input)
    data_set = np.zeros((num_of_test, patch_side, patch_side, patch_side))
    for i in trange(num_of_test):
        data_set[i, :] = np.reshape(io.read_mhd_and_raw(list[i]), [patch_side, patch_side, patch_side])
    return data_set

def plt_loss(epochs, train_loss_list, val_loss_list, outdir):
    # plot graph
    plt.figure()
    plt.plot(range(epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.grid()
    plt.savefig(outdir + "loss.png")

def add_noise(img, device):
    # add random noise to torch tensor
    noise_factor = 0.4
    noise = torch.normal(mean=0, std=0.5, size=img.size(), generator=torch.manual_seed(1)).to(device)
    noisy_img = img + noise*noise_factor
    # Clip the images to be between 0 and 1
    noisy_img[noisy_img>=1.] = 1.
    noisy_img[noisy_img<=0.] = 0.
    return noisy_img