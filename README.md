# An Adjacency-Adaptive Gaussian Process Method for Sample Efficient Response Surface Modeling and Test-Point Acquisition
By Stanford Martinez and Adel Alaeddini


## Intro
This repository contains the code for running two examples of simulations as seen in the manuscript (titled above). The specific examples are for the `(1) Qing (3D)` and `(2) Cosine (10D)` functions in Figure 3 therein.

## Recommended Installation
The steps below are listed to allow users to configure a python environment similar to that used by the authors of this manuscript to produce examples of output as seen in Figure 3 therein.

This is our prescribed method for running this code, and has been tested on `Windows`. To prevent dependency issues with users' existing python installations, our code was designed to perform the following `by running the "Figure 3.py" file`:

1) utilize the version of python executing the code
2) create a virtual environment (`venv`) based on the utilized version of python
3) install all necessary packages in `venv` using the `requirements.txt` file
4) create a temporary python file titled `temp_exe.py`, used to run the simulation in the `venv`
5) run the simulation in `venv`

### Setup and Running
1) Download this branch (`main`) as a zip file
2) Extract all contents into some folder `example: C:\Users\your user name\downloads\aagp_demo`
    - if you perform "extract all" and extract the folder in the zip file as well, you can extract the folder to: 

        `example: C:\Users\your user name\downloads\`
        
        and your path will be:
        
        `example: C:\Users\your user name\downloads\aagp-demo`
----
- If you already have a dedicated environment (with `git` installed), please activate the environment and skip to step `(7)`. Please see the `Notes` section for more information on versioning and compatibility.
----
3) install anaconda and run the `anaconda PowerShell prompt` application, then enter the following commands:
4) `conda create --name aagp_demo python=3.9 -y`
5) `conda install -y git` (we utilize anaconda due to streamline installing git here)
6) `conda activate aagp_demo`
7) `cd "C:\Users\your user name\downloads\aagp_demo"`
   - if you perform "extract all" and extract the folder in the zip file as well, you can `cd` the folder to:
   
       `example: C:\Users\your user name\downloads\aagp-demo` 
8) `python "Figure 3.py"`

The code will then execute and perform the procedure outlined in the beginning of this section.



## Simulation & Runtime Notes
### Changing the Example to Test
1) open the `Figure 3.py` file
2) Change the value of `test_function` keyword argument in the declaration of the `RUN_PIPELINE` function (`line 8`):
  - `test_function = 0` to run a simulation for the `Qing (3D)` function
  - `test_function = 1` to run a simulation for the `Cosine (10D)` function
### Runtime Notes
- Once started after step `(8)` above, the code will then proceed to install packages and run the simulation.
- At the end, it will output `Example Output.jpg` to the directory in which `EXECUTOR.py` is placed.
- The DeepGP methodolgy is very memory-intensive and may cause OOM errors running in parallel.
- Runtime is roughly ~100min on an intel 13900HX processor with 28 cores and 96GB RAM running in parallel for the `Qing (3D)` example, and ~360min for the `Cosine (10D)` Example.
- This large difference in compute time reported for the `Cosine (10D)` function is primarily attributed to the DeepGP competitor, the layers and dimensions used for its configuration, the dimensionality of the dataset, and 2,000 iterations (as is used by the model in our manuscript) for DeepGP's training.
- The `Qing (3D)` Example takes a shorter amount of time due to similar results as seen in the manuscript being achievable with using only 200 training iterations instead of 2,000.



## Python Versioning Notes
- We have tested this code on python versions `3.7-3.12` and have found that the `Gpy` package, leveraged by the Deep Gaussian Process model used as a comparison framework (https://github.com/SheffieldML/PyDeepGP), requires the installation of `Microsoft Visual C++ 14.0` to resolve compiler issues with python versions `3.10`, `3.11`, and `3.12`. Please refer to the `Visual C++ Installation` section for more details.
- For specifics on python versions referenced by the authors of the DeepGP package, please see `line 37` at the following linked file (other specifics are available in the linked file): https://github.com/SheffieldML/PyDeepGP/blob/master/setup.py#L37
        
- We leverage the usage of `Anaconda` for running this code. Noted in the `Setup and Running` section, it can streamline the process for installing environments and `git`, a version control system that will install the Deep Gaussian Process comparison model from the github link above.
- If you do not use `Anaconda` and have your own setup, please ensure you have `git` installed and usable with the python environment you designate for use.


## Visual C++ Installation (For `Python 3.10` and Higher)
For python versions `3.10` and onwards, `Visual C++ 14.0` and higher are required for compilers. If you have python versions `3.7`, `3.8`, or `3.9`, this step may not be required.
- use this link to download Microsoft C++ Build Tools:
        
        https://visualstudio.microsoft.com/visual-cpp-build-tools/

- Click on `download build tools` in the top left
- Open the application and proceed until you reach the menu of installation selections
- Ensure `Desktop development with C++` is checked, and allow additional items to be installed if they are checked along with the aforementioned item.
- Click `install` in the bottom-right, and allow the application to proceed.
- Once complete, you may close the window(s) for the installer.
- Restart your python, and re-try the `Setup and Running` section.
