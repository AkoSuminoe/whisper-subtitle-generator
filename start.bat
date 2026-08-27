@echo off
setlocal enabledelayedexpansion

REM Force a UTF-8 console. The default cp1252 codepage raises UnicodeEncodeError
REM on Turkish characters (such as a dotless i) appearing in any traceback.
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "MARKER=%VENV_DIR%\.deps_installed"

echo ==================================
echo  Whisper Subtitle Generator
echo ==================================
echo.

REM --- 1. Python -------------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Install Python from https://www.python.org/downloads/ and re-run this script.
    pause
    exit /b 1
)

REM --- 2. Virtual environment -------------------------------------------------
if not exist "%VENV_PY%" (
    echo [setup] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"

REM --- 3. Dependencies (skipped once the marker file exists) ------------------
REM Written with goto rather than a parenthesised if-block on purpose: the commands
REM below contain parentheses, which cmd's block parser handles badly.
if exist "%MARKER%" goto deps_ready

echo [setup] First run: installing dependencies. This downloads several GB.
echo.

python -m pip install --upgrade pip
if errorlevel 1 goto install_failed

echo.
echo [setup] Installing PyTorch from the CUDA 12.8 index...
echo         Required for Blackwell / sm_120. The default PyPI build does not
echo         support this GPU and would silently fall back to the CPU.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto install_failed

echo.
echo [setup] Installing Whisper and CustomTkinter...
python -m pip install -U openai-whisper customtkinter
if errorlevel 1 goto install_failed

echo.
echo [setup] Verifying the GPU build...
python -c "import torch; print('torch', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

>"%MARKER%" echo installed
echo.
echo [setup] Dependencies installed.
echo.

:deps_ready

REM --- 4. ffmpeg / ffprobe ----------------------------------------------------
call :check_ffmpeg
if "%FFMPEG_OK%"=="1" goto ffmpeg_ready

echo [setup] ffmpeg was not found. Installing it with winget...
winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
echo.

call :check_ffmpeg
if "%FFMPEG_OK%"=="1" goto ffmpeg_ready

REM winget cannot refresh PATH for an already-running shell, so locate the install
REM directly and prepend it for this session.
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do (
    for /d %%B in ("%%~fD\ffmpeg-*") do (
        if exist "%%~fB\bin\ffmpeg.exe" set "PATH=%%~fB\bin;!PATH!"
    )
)

call :check_ffmpeg
if "%FFMPEG_OK%"=="1" (
    echo [setup] Using ffmpeg from the winget install folder for this session.
    goto ffmpeg_ready
)

echo.
echo [ERROR] ffmpeg and ffprobe are still unavailable.
echo Whisper cannot decode audio without them.
echo Install manually, then re-run this script:
echo     winget install --id Gyan.FFmpeg -e
echo.
echo If you just installed it, close this window and open a new one so that
echo Windows picks up the updated PATH.
pause
exit /b 1

:ffmpeg_ready

REM --- 5. Launch --------------------------------------------------------------
echo [run] Starting Whisper Subtitle Generator...
echo.
python app.py
if errorlevel 1 (
    echo.
    echo [ERROR] The app exited with an error. See the traceback above.
    pause
)
exit /b 0

:install_failed
echo.
echo [ERROR] Dependency installation failed. See the messages above.
echo The marker file was not written, so the next run will retry the install.
pause
exit /b 1

REM Sets FFMPEG_OK=1 only when both tools resolve on PATH.
:check_ffmpeg
set "FFMPEG_OK="
where ffmpeg >nul 2>&1 || goto :eof
where ffprobe >nul 2>&1 || goto :eof
set "FFMPEG_OK=1"
goto :eof
