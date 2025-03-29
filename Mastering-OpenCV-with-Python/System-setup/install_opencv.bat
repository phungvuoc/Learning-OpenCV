@echo off
REM Check if run as Administrator
net session >nul 2>&1
if %errorLevel% == 1 (
    echo This script requires Administrator privileges.
    pause
    exit
)

REM Set the download folder (Downloads folder of the current user)
set DOWNLOAD_FOLDER=%UserProfile%\Downloads

REM Download Miniconda Installer for Python 3.11 to the Downloads folder
echo Downloading Miniconda Installer for Windows...
powershell -command "Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Windows-x86_64.exe"

REM Install Miniconda silently from the Downloads folder
echo Installing Miniconda...
%DOWNLOAD_FOLDER%\Miniconda3.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S /D=%UserProfile%\Miniconda3

REM Check if Miniconda installed successfully
if exist "%UserProfile%\Miniconda3" (
    echo Miniconda installation successful.
) else (
    echo Miniconda installation failed. Exiting...
    exit /b 1
)

REM Initialize Conda
echo Initializing Miniconda...
call %UserProfile%\Miniconda3\Scripts\activate.bat
call conda init

REM Create a new Conda environment with Python 3.11
echo Creating a new Conda environment with Python 3.11...
call conda create --name opencv-env python=3.11 -y

REM Activate the new environment
echo Activating the new environment...
call conda activate opencv-env

REM Verify Python installation
echo Verifying Python installation...
python --version

REM Upgrade pip to the latest version
echo Upgrading pip to the latest version...
pip install --upgrade pip

REM Download requirements.txt to the Downloads folder
echo Downloading requirements.txt...
powershell -command "Invoke-WebRequest -Uri 'https://www.dropbox.com/scl/fi/aetjx5e8dd1ssy7ftu71a/requirements.txt?rlkey=dioa1fx6o7tcgk2nosx6jpigs&st=tmvu7nkg&dl=1' -OutFile %DOWNLOAD_FOLDER%\requirements.txt"

REM Check if requirements.txt was downloaded successfully
if exist "%DOWNLOAD_FOLDER%\requirements.txt" (
    echo requirements.txt downloaded successfully.
) else (
    echo Failed to download requirements.txt. Exiting...
    exit /b 1
)

REM Automatically install OpenCV and other required libraries from the downloaded requirements file
echo Installing OpenCV and other required libraries...
pip install -r %DOWNLOAD_FOLDER%\requirements.txt

REM Show completion message and wait for user to press a key before exiting
echo Installation complete. You can now use Python with OpenCV!
pause
