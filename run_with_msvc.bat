@echo off
REM Launch Python inside a MSVC + CUDA developer environment.
REM Calls vcvarsall.bat (VS 2022 BuildTools) then runs the script passed as %1.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Add venv Scripts to PATH so ninja.exe is reachable
set "PATH=D:\Personal Projects\Little Bit\.venv\Scripts;%PATH%"

REM Set CUDA arch target
set TORCH_CUDA_ARCH_LIST=8.6

cd /d "D:\Personal Projects\Little Bit"

"D:\Personal Projects\Little Bit\.venv\Scripts\python.exe" %*
