@echo off

set anaconda=C:/ProgramData/Anaconda3/envs/tl/python.exe
set py=E:/git/DAE/main.py

set indir=E:/git/pytorch/vae/input/s0/filename.txt

call %anaconda% %py% --input %indir% --topo True --ramda 100000 --model "E:\git\DAE\results\artificial\z_24\B_0.1\R_0\model.pkl" --epochs 50

pause