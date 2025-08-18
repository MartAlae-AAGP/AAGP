# An Adjacency-Adaptive Gaussian Process Method for Sample Efficient Response Surface Modeling and Test-Point Acquisition
----
By Stanford Martinez and Adel Alaeddini

## Intro
This repository contains the code for running a simulation to output the subplot pertaining to the `Qing (3D)` test function in `Figure 3` of the manuscript.

----

## Setup & Code Execution
The steps below are listed to allow users to configure a python environment similar to that used by the authors of this manuscript. As mentioned the resultant output will be a single subplot of the `Qing (3D)` test function. We have tested the code to run on `windows`, `Windows Subsystem for Linux (WSL)`, `Ubuntu Linux` via VirtualBox virtual machine, and `MAC-OS`. We provide instructions to assist with setting up and running the simulation with and without anaconda/miniconda configurations.

### Windows (Without Anaconda)
- Please skip to step (5) if you have a dedicated environment with git installed

1) Download this branch (`main`) as a zip file
2) Extract all contents into some folder (example: `C:\Users\YourUserName\Downloads\AAGP-main`)
3) Install `Python`
    1)  Go to `https://www.python.org/downloads/release/python-3913/` and download the 64-bit installer
    2)  Run the installer:
        1) Check `Add Python 3.9 to PATH`
        2) Click `Customize Installation` and make sure `pip`, `venv`, and `Add Python to environment variables` are selected
        3) Complete the installation
4) Install `Git` on Windows (Recommended)
    1) Download Git for Windows: `https://git-scm.com/download/win`
    2) Run the installer:
        1) Accept all defaults
        2) Be sure to select `Git from the command line and also from 3rd-party software` to add Git to your PATH so `pip` can use it
    3) Finish setup
5) Open `Command Prompt` (not powershell), Navigate to the designated folder for windows, and run the python file:
    1) `cd "C:\Users\YourUserName\Downloads\AAGP-main\AAGP-main"`
    2) `py -3.9 "Figure3.py"`
6) When the code completes, it will output `Figure 3.jpg` which can be opened via the file explorer.

### Windows (With Anaconda)
- Please skip to step (7) if you have a dedicated environment with git installed, and have the environment active and ready

1) Download this branch (`main`) as a zip file
2) Extract all contents into some folder (example: `C:\Users\YourUserName\Downloads\AAGP-main`)
3) Install anaconda and create a new `python 3.9` environment via `Anaconda Powershell Prompt` 
    - example code: `conda create --name aagp_demo python=3.9 -y`
4) Activate the new environment: `conda activate aagp_demo`
5) Navigate to the `Windows` folder where the code was extracted: `cd "C:\Users\YourUserName\Downloads\AAGP-main\AAGP-main"`
6) Install git: `conda install -y git`
7) Run the code in the directory: `python "Figure3.py"`
8) When the code completes, it will output `Figure 3.jpg` which can be opened via the file explorer.

### UBUNTU Linux OS (Without Anaconda)
- Please skip to step (5) if you have a dedicated environment with git installed and proper permissions

1) After downloading the repo zip file and extracting to a directory (for this example: `/home/vboxuser/Downloads/AAGP-main/AAGP-main`), open a `terminal` and navigate to it:
    1) `cd /home/vboxuser/Downloads/AAGP-main/AAGP-main`
2) install python 3.9
    1) `sudo apt update`
    2) `sudo apt install -y software-properties-common`
    3) `sudo add-apt-repository ppa:deadsnakes/ppa`
    4) `sudo apt update`
    5) `sudo apt install -y python3.9 python3.9-venv python3.9-distutils`
3) install git
    1) `sudo apt update`
    2) `sudo apt install -y git`
4) you may need to give read/write permissions in the current folder:
    1) `chmod u+rwx .`
5) run the code in the directory
    1) `python3.9 "Figure3.py"`
6) once the code completes, the image `Figure 3.jpg` will be output. it can be opened in an image viewer you have available.

### UBUNTU Linux OS (With Anaconda)
- Please skip to step (6) if you have a dedicated environment with git installed and proper permissions

1) After downloading the repo zip file and extracting to a directory (for this example: `/home/vboxuser/Downloads/AAGP-main/AAGP-main`), open a `terminal` and navigate to it:
    1) `cd /home/vboxuser/Downloads/AAGP-main/AAGP-main`
2) allow the folder permissions to read/write
    1) `chmod u+rwx .`
3) Download miniconda and install it:
    1) `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    2) `bash Miniconda3-latest-Linux-x86_64.sh`
    3) Proceed through the prompts and enter `yes` when asked to initialize conda when starting a terminal
    4) Close and reopen your terminal to activate conda.
4) Create an environment, activate it, and install git
    1) `conda create --name aagp_demo python=3.9 -y`
    2) `conda activate aagp_demo`
    2) `sudo apt update`
    3) `sudo apt install -y git`
5) If this is your first time installing conda newly and see an error: `CondaToSNonInteractiveError`, enter the following commands:
    1) `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
    2) `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r`
6) Navigate to the directory and run the python file:
    1) `cd /home/vboxuser/Downloads/AAGP-main/AAGP-main`
    2) `python "Figure3.py"`
7) once the code completes, the image `Figure 3.jpg` will be output. it can be opened in an image viewer you have available.


### WSL Linux (Without Anaconda)
- Please skip to step (4) if you have a dedicated environment with git installed

1) After downloading the repo zip-file and extracting to your directory (example: `C:\Users\YourUserName\Downloads\AAGP-main\AAGP-main`), mount it via the ubuntu terminal:
    1) `cd ~`
    2) `cp -r "/mnt/c/Users/YourUserName/Downloads/AAGP-main/AAGP-main" aagp_demo`
       - please note the quotes around the target directory
    4) `cd aagp_demo`
2) install python 3.9
    1) `sudo apt update`
    2) `sudo apt install -y software-properties-common`
    3) `sudo add-apt-repository ppa:deadsnakes/ppa`
    4) `sudo apt update`
    5) `sudo apt install -y python3.9 python3.9-venv python3.9-distutils`
3) install eog and git (if not installed):
    - `sudo apt install eog`
    - `sudo apt install -y git`
4) run the code using the installed version of python:
    1) `cd ~/aagp_demo`
    2) `python3.9 "Figure3.py"`
5) Once complete, use eog to open the figure (it may take a few moments as it is a high-resolution photo):
    - `eog "/home/YourUserName/aagp_demo/Figure 3.jpg"` or `eog "Figure 3.jpg"`

### WSL Linux (With Anaconda)
- Please skip to step (4) if you have a dedicated environment with git installed

1) Install conda in WSL:
    1) `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    2) `bash Miniconda3-latest-Linux-x86_64.sh`
    3) proceed through the instructions (accept the install path and select `yes` to allow it to `initialize miniconda on startup`)
    4) Close and reopen your WSL terminal to activate conda
2) Create the environment, activate it, and install Git and eog
    1) `conda create --name aagp_demo python=3.9 -y`
    2) `conda activate aagp_demo`
    3) `sudo apt update`
    4) `sudo apt install -y git`
    5) `sudo apt install eog`
3) If this is your first time installing conda newly and see an error: `CondaToSNonInteractiveError`, enter the following commands:
    1) `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
    2) `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r`
    3) retry step (2)
4) Access the folder with your code (example: `C:\Users\YourUserName\Downloads\AAGP-main\AAGP-main`), mount it via the ubuntu terminal:
    1) `cd ~`
    2) `cp -r "/mnt/c/Users/YourUserName/Downloads/AAGP-main/AAGP-main" aagp_demo`
       - please note the quotes around the target directory
    3) `cd aagp_demo`
5) From within that folder, run the simulation and open the figure when it completes:
    1) `python3.9 "Figure3.py"`
    2) `eog "/home/YourUserName/aagp_demo/Figure 3.jpg"` or `eog "Figure 3.jpg"`
        - it may take a few moments as it is a high-resolution photo
    
### MAC-OS (Without Anaconda/Miniconda)
- This setup assumes you have a compatible version of python (`3.9` recommended) installed arleady.
- Please skip to step (4) if you have a dedicated environment with git installed.

1) Download this repo and extract to some desired folder (example: extracting in `~/Downloads` gives `~/Downloads/AAGP-main`)
2) Open a new terminal and navigate to the folder: `cd ~/Downloads/AAGP-main`
3) Reset the environment by deleting previous runs (if desired), creating a new environment, and activating it:
    1) `rm -rf aagp_env`
    2) `python3.9 -m venv aagp_env`
    3) `source aagp_env/bin/activate`
    4) `cd ~/Downloads/AAGP-main`
4) Manually install `pyDeepGp`
    1) `git clone https://github.com/SheffieldML/pyDeepGP.git`
5) Run the code (navigate to the folder and activate the environment if needed):
    1) if needed: `source aagp_env/bin/activate`
    2) if needed: `cd ~/Downloads/AAGP-main`
    3) `python "Figure3_MAC.py"`
6) When the code completes, it will output `Figure 3.jpg` which can be opened via the file explorer.

- Note:
    - if the terminal is closed you can reopen it and start from step (3).
    - if you do not have git installed, you may be prompted to install it. Please do so and go back to step 4.

### MAC-OS (With Miniconda)
- This setup assumes you have miniconda or anaconda installed arleady.
- Please skip to step (4) if you have a dedicated environment with git installed.

1) Download this repo and extract to some desired folder (example: extracting in `~/Downloads` gives `~/Downloads/AAGP-main`)
2) Open a new terminal and navigate to the folder: `cd ~/Downloads/AAGP-main`
3) Create and activate an environment with python 3.9, activate it, and manually install `pyDeepGP`:
    1) `conda create -n aagp_demo python=3.9 -y`
    2) `conda activate aagp_demo`
    3) `git clone https://github.com/SheffieldML/pyDeepGP.git`
4) If you have not done so, activate the environment, navigate to the folder in step 2, and run the python file: 
    1) `conda activate aagp_demo`
    2) `cd ~/Downloads/AAGP-main`
    3) `python "Figure3_MAC.py"`
5) When the code completes, it will output `Figure 3.jpg` which can be opened via the file explorer.
- Note:
    - if the terminal is closed you can reopen it and start from step (4).
    - if you do not have git installed, you may be prompted to install it. Please do so and go back to step 4.

----

## Additional Notes
### Hardware Requirements
The `DeepGP` competitor is computationally intensive and dominates the runtime and memory usage of this simulation.
- Minimum Recommended:
    - CPU: 8+ cores
    - RAM: 16 GB (32 GB strongly recommended)
- Tested System:
    - Intel i9-13900Hx (28 cores employed)
    - 96 GB Ram
    - Runtime: ~180 minutes in parallel execution
    
Users running on less powerful systems should expect longer runtimes, particularly due to the `DeepGP` competitor.

### Runtime Notes
- Once started, the code will then proceed to install packages and run the simulation.
- At the end, it will output `Figure 3.jpg` to the directory in which `Figure3.py` or `Figure3_MAC.py` is placed.
- The DeepGP methodolgy is very memory-intensive and may cause OOM errors running in parallel.
- Runtime is roughly ~180min on an intel 13900HX processor with 28 cores and 96GB RAM running in parallel.
- The primary hardware stressor arises from the DeepGP competitor, which imposes heavy demands on both memory and CPU. As a result, some low-end systems may yield long run-times due to computational complexity.

### Python Versioning Notes
We have tested this code on python versions `3.7-3.12` and have found that the `GPy` package is compatible with `3.7`, `3.8`, and `3.9` versions before compiler issues occured due to code differences in python past 3.9 regarding compilers. This repo and code execution is mostly compatible with the earlier versions of python, with some `patch` versions, for example python `3.9` patch `0` or  `3.9.0` being incompatible, while `3.9.13` and `3.9.23` are compatible. As such, we recommend leveraging the above instructions and installing the latest version of `3.9` to avoid compatibility issues.
