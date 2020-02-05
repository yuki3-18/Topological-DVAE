@echo off

set anaconda=C:/ProgramData/Anaconda3/envs/tl/python.exe
set py=E:/git/DAE/eval.py

set indir=E:/git/pytorch/vae/input/s0/filename.txt

set outdir=E:\git\DAE\results\artificial\z_24\B_0.1\R_0\
set outdir1=E:\git\DAE\results\artificial\z_24\B_0.1\R_1.0\
set outdir2=E:\git\DAE\results\artificial\z_24\B_0.1\R_2500.0\
set outdir3=E:\git\DAE\results\artificial\z_24\B_0.1\R_10000.0\
set outdir4=E:\git\DAE\results\artificial\z_24\B_0.1\R_1000.0\

::call %anaconda% %py% --input %indir% --outdir %outdir%
call %anaconda% %py% --input %indir% --outdir %outdir1%
::call %anaconda% %py% --input %indir% --outdir %outdir2%
::call %anaconda% %py% --input %indir% --outdir %outdir3%
::call %anaconda% %py% --input %indir% --outdir %outdir4%

pause