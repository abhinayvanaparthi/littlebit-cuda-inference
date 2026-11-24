@echo off
REM Launch Python inside an MSVC + CUDA developer environment.
REM Calls vcvarsall.bat (VS 2022 BuildTools / Community) then runs the script passed with arguments.

set "SCRIPT_DIR=%~dp0"

REM Allow user to override VS_VCVARSALL if installed in a custom location
if "%VS_VCVARSALL%"=="" (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS_VCVARSALL=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS_VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS_VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS_VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    ) else (
        echo WARNING: vcvarsall.bat not found automatically. Set the VS_VCVARSALL environment variable to your Visual Studio vcvarsall.bat path.
    )
)

if defined VS_VCVARSALL (
    call "%VS_VCVARSALL%" x64
)

REM Add venv Scripts to PATH so ninja.exe and compiler tools are reachable
set "PATH=%SCRIPT_DIR%.venv\Scripts;%PATH%"

REM Set CUDA arch target
set TORCH_CUDA_ARCH_LIST=8.6

cd /d "%SCRIPT_DIR%"

"%SCRIPT_DIR%.venv\Scripts\python.exe" %*
