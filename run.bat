@echo off
cd /d %~dp0

set anaconda=C:/ProgramData/Anaconda3/envs/tl/python.exe
set py=./main.py

call %anaconda% %py% --topo

pause