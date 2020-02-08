# Topological-DVAE
An implementation of Denoising-Variational AutoEncoder with Topological loss

## Requirements
```
1. PyTorch
2. SimpleITK
3. tqdm
4. visdom
5. TopologyLayer
6. Gudhi
```

## Usage
Training Model: main.py\
Reconstructing Image: predict_gen.py\
Check : predict_spe.py

## Reference
[1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes, (Ml), 1–14. https://doi.org/10.1051/0004-6361/201527329

[2] James R. Clough, et al. (2019). "A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology". https://arxiv.org/abs/1910.01877

[3] https://github.com/pytorch/examples/tree/master/vae

[4] https://github.com/JamesClough/topograd

[5] https://github.com/bruel-gabrielsson/TopologyLayer

