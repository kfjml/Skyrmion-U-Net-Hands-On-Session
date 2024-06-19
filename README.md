[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kfjml/Skyrmion-U-Net-Hands-On-Session/HEAD?labpath=Editor.ipynb)

# Welcome to the Skyrmion U-Net hands-on repository !

The dataset in this repository is an extraction of the following Zenodo repository (v2.0) from Winkler et al.: https://doi.org/10.5281/zenodo.10997175. This repository is intended for getting started with the Skyrmion U-Net. The authors of this repository are Kilian Leutner, Thomas Brian Winkler, Isaac Labrie-Boulay, Alena Romanova, Hans Fangohr, Mathias Kläui, Raphael Gruber, Fabian Kammerbauer, Klaus Raab, Jakub Zazvorka, Elizabeth Martin Jefremovas and Robert Frömter.

For any questions regarding this repository, please contact Kilian Leutner at kileutne@students.uni-mainz.de.

## Getting started

In the following Sections 1-5, the installation and execution of the required software (installation of a Python environment in Anaconda and starting Jupyter Notebooks) are explained to enable participation in the hands-on session. It explains how to install the necessary components for both CPU and GPU execution, although installation for the CPU alone is sufficient for the session.

In Section 6, it is explained how you can also run these Jupyter Notebooks online without installing anything on your device. Please only do this if you are unable to manage the local installation. The servers that enable this online execution have limited resources, and therefore running them during the session might not work. For details, refer to this section.

In Section 7, you will find more detailed information about the installed packages and a guide for complete manual installation.

### 1 Download this repository

If you do not already have this repository, download it. You can do this by using 

```
git clone https://github.com/kfjml/Skyrmion-U-Net-Hands-On-Session
```

if you have `git` installed on your device. If you do not have `git`, navigate to the main page of the GitHub repository [https://github.com/kfjml/Skyrmion-U-Net-Hands-On-Session](https://github.com/kfjml/Skyrmion-U-Net-Hands-On-Session) (which you are likely already on if you are reading this readme) and click on the green button labeled `Code`, then click on `Download ZIP`.

Unzip this archive and open a command line. Navigate to this unzipped folder.

### 2 Install & start miniconda

Miniconda is a lightweight Python distribution that includes package management `conda`, simplifying the installation of Python packages. Install the newest version of Miniconda if you don't have it already (if you have Anaconda, you can also use that). Details can be found at:

[https://docs.anaconda.com/free/miniconda/](https://docs.anaconda.com/free/miniconda/)

[https://docs.anaconda.com/free/miniconda/miniconda-install/](https://docs.anaconda.com/free/miniconda/miniconda-install/)

Afterward, please start miniconda (or anaconda) from the command line (start miniconda/anaconda prompt on windows) and navigate to the directory of this repository where this README.md is located.

### 3 Setup of the skyrmion U-Net environment 

Installation on the GPU can be more challenging depending on the system. For the hands-on session, the CPU environment is completely sufficient. However, for higher performance and to train large Skyrmion U-Nets, it is recommended to install the GPU version. You can install both environment s for CPU and GPU independently of each other, and also both together.

The following explains the automatic installation of the environment. In Section 7, you will find an explanation of the installed packages and a guide of a manual installation.

#### 3.1 Installation of the CPU environment 

When you want to run the Skyrmion U-Net only on the CPU, execute the following command to create a conda environment `skyrmion_unet_cpu` for CPU usage with all necessary packages:

```
conda env create -f environment.yml
```

`environment.yml` is a YAML file where the environment, along with its packages and the specific versions recommended for use, is defined.

In the case that this installation **does not** work and returns an error (which may occur, for example, with Mac M1/M2 processors), please try the following command (**do not** execute if the previous environment installation succeeded):

```
conda env create -f environment_cpu_v2.yml
```

`environment_cpu_v2.yml` is also a YAML file, but in this case, the TensorFlow package version (important package, details see Section 7.1) is not specified. It will automatically search for a suitable package version. It's possible that during the previous installation attempt, the `skyrmion_unet_cpu` environment was already created. In this case, before this new installation can proceed successfully, the environment must be deleted. This process is explained in Section 5. If this second installation also **does not** work, try the following command (**do not** execute if either of the previous environment installation commands succeeded):

```
conda env create -f environment_cpu_v3.yml
```

`environment_cpu_v3.yml` is also a YAML file, but none of the various Python packages have a specified version. Suitable versions of the packages for the device will be automatically searched. Here, it might also be necessary to delete the environment again before this command works, for details see Section 5.

#### 3.2 Installation of the GPU environment 

##### Preliminary remarks for the GPU installation

If you intend to run the Skyrmion U-Net on the GPU, you need to install the NVIDIA GPU driver, the CUDA Toolkit, and cuDNN SDK. **Before** installing any of these, please refer to [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip) for detailed information, as the Skyrmion U-Net is based on TensorFlow. Only follow the steps up to `Step 3: Install TensorFlow` as the subsequent steps are explained in this README. As explained in this installation guide, the current GPU version of TensorFlow does not work directly on Windows. To install the GPU environment on Windows, you need to use Windows Subsystem for Linux (WSL) (Ubuntu as the distribution in WSL is probably the best choice for these purposes). You can test the driver's functionality using `nvidia-smi`.

##### Installation of the GPU environment 

When you want to run the skyrmion U-Net exclusively on the GPU, run the following command which will create a conda environment `skyrmion_unet_gpu` for the GPU usage with all necessary packages

```
conda env create -f environment_gpu.yml
```

`environment_gpu.yml` is a YAML file where the environment, along with its packages and the specific versions recommended for use, is defined.

In the case that this installation **does not** work and returns an error (which may occur, for example, with Mac M1/M2 processors), please try the following command (**do not** execute if the previous environment installation succeeded):

```
conda env create -f environment_gpu_v2.yml
```

`environment_gpu_v2.yml` is also a YAML file, but in this case, the TensorFlow package version (important package, details see Section 7.1) is not specified. It will automatically search for a suitable package version. It's possible that during the previous installation attempt, the `skyrmion_unet_gpu` environment was already created. In this case, before this new installation can proceed successfully, the environment must be deleted. This process is explained in Section 5. If this second installation also **does not** work, try the following command (**do not** execute if either of the previous environment installation commands succeeded):

```
conda env create -f environment_gpu_v3.yml
```

`environment_gpu_v3.yml` is also a YAML file, but none of the various Python packages have a specified version. Suitable versions of the packages for the device will be automatically searched. Here, it might also be necessary to delete the environment again before this command works, for details see Section 5.

### 4 Using the skyrmion U-Net

To activate the environment, you use the environment name of the environment  you installed. You activate the environment using the following command and the associated environment name, in the following command `skyrmion_unet_cpu` (the standard name for the CPU environment):

```
conda activate skyrmion_unet_cpu
```

Afterwards start a jupyter notebook with

```
jupyter notebook
```

This command will provide you with a localhost address, which you can then open in your web browser to access the Jupyter Notebook. Then you can try out the Jupyter notebooks for prediction (`Prediction_tutorial.ipynb`) and training (`Training_tutorial.ipynb`) in the main folder of the repository, and also create your own notebooks with own code for the Skyrmion U-Net. 

In these notebooks, please execute the cells sequentially from top to bottom (execution of a cell: when a cell is selected, press shift+enter or click the run button (play button symbol)). In the prediction notebook, you can interactively modify the outputs using the displayed GUI. If you want to change these cells with a GUI again after executing other cells, please execute these cells again.

The prediction notebook (`Prediction_tutorial.ipynb`) is designed to work completely with execution on a regular CPU. In the training notebook (`Training_tutorial.ipynb`), you can also execute on the CPU in the first part; it is designed to train a small U-Net, which can also be trained on the CPU. For the second part, a large U-Net is trained on a large dataset. A GPU is recommended for this, but the focus of the hands-on session does absolutely not lie on this second part of the training notebook.

### 5 Deleting installed environments

If you no longer want to have the installed Skyrmion U-Net environments after the Hands-On session, or if there were any issues during the installation process and you want to remove the environment to install the environment with the same name (the standard name `skyrmion_unet_cpu` is used for the CPU environment, and `skyrmion_unet_gpu` is used for the GPU environment, in the YAML files) again with another YAML file, execute the following command along with the associated environment name, which is in the following command `skyrmion_unet_cpu` (the standard name for the CPU environment):

```
conda remove -n skyrmion_unet_cpu --all
```

During deletion, you will be prompted to confirm the removal of certain files and folders. Please press `y` here.

### 6 Notebooks online ausführen

Please only do this if you are unable to manage the local installation. The servers that enable this online execution have limited resources, and running them during the session might not work.

#### 6.1 Binder

You can open this repository online using the BinderHub from `mybinder.org`, view the notebooks, and execute them. The `Prediction_tutorial.ipynb` notebook, which demonstrates the prediction of Skyrmion U-Nets and allows for prediction on own Kerr images, works. In the `Training_tutorial.ipynb` notebook, cells can be executed, but training itself does not work because the resources provided by `mybinder.org` are insufficient for that purpose. At` mybinder.org`, only execution on CPU is possible.

It may take a while for the repository to load and become executable after clicking on the corresponding link for this repository. `Prediction_tutorial.ipynb` will open automatically. You can also open the `Training_tutorial.ipynb` notebook in MyBinder (button on the left sidebar). If there is too much traffic on MyBinder, execution may not be possible. Therefore, only use this tool during the hands-on session if you cannot manage the installation yourself, to give others who cannot manage the installation the opportunity to use Binder. Outside of the hands-on session, feel free to use MyBinder, where executing and making predictions on your own data should work; the corresponding notebook `Prediction_tutorial.ipynb` will, as mentioned earlier, open automatically.

To open this repository on `mybinder.org`, please click the following button: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kfjml/Skyrmion-U-Net-Hands-On-Session/HEAD?labpath=Editor.ipynb) 

#### 6.2 Google Colab

Google Colab by Google offers more computer resources available for the online execution of notebooks in the background compared to Binder. A major advantage is that both CPU and powerful GPUs are available, depending on the resources available on the Colab servers. However, the runtime is limited to approximately 2 hours, and at the beginning of the runtime, packages need to be installed (which happens automatically when executing the first cell, but it takes some time). The resources on the CPU also depend on the resources currently available on the Colab server. Additionally, a Google account is required.

To execute the Jupyter Notebooks needed in the session, please follow these steps:

Open [https://colab.research.google.com/](https://colab.research.google.com/) (and sign in with your Google account if you haven't already).

Afterwards, open a notebook (the dialog appears automatically or click on File -> Open Notebook) and select `GitHub` in the dialog. Then, provide the URL of this repository as the `GitHub URL`, which is `https://github.com/kfjml/Skyrmion-U-Net-Hands-On-Session`, and select one of the two notebooks, `Prediction_tutorial.ipynb` (the slightly more important notebook) or `Training_tutorial.ipynb`. After that, the notebook opens.

Select `Runtime` -> `Change runtime type` and choose `T4 GPU`. This will attempt to execute the session on the GPU. It may happen that later on, you receive a message that the GPU is not available. In this case, in the free version, which is completely sufficient for the hands-on session, the execution will be done only on the CPU, although execution on the GPU is much faster (the notebooks have been designed so that execution on the CPU should also work). Now, when you execute the first cell, the runtime begins (which can last approximately 2 hours). You **must** execute this cell at the beginning or just before the start of the hands-on session because important packages will be installed. (for the execution of a cell: when a cell is selected, press shift+enter or click the run button (play button symbol)). The installation takes some time, and after that, you need to restart the session (which is also indicated to you). After restarting the session, you must execute the first cell again. Then, you can also execute the other cells.

You need to perform each step for both Jupyter Notebooks (`Prediction_tutorial.ipynb` and `Training_tutorial.ipynb`) shortly before the start of the hands-on session (or if you want to execute these notebooks via Colab outside the hands-on session, what you are welcome to do).

If you have access to the `T4 GPU`, you can execute the second part of `Training_tutorial.ipynb`, where a large U-Net is trained on a large dataset. For this, a large GPU is required, and the `T4 GPU` is perfectly suitable for this task. This is also a practical tool if you want to train a large network on your own large dataset after the hands-on session and do not have a GPU device available.

### 7 Further informations

#### 7.1 Explanation of the packages installed in the environment

`pip` is a Python package management system that is used together with Anaconda/Miniconda in the environment. `tensorflow` is the package used for machine learning and neural networks, and the `albumentations` package is used for data augmentation during training. The `matplotlib` package is used for plotting data, especially for MOKE images and their predictions with the Skyrmion U-Net. Furthermore, the `pandas` package is used for data saving, analysis, and editing. The `chardet` package, which is a Universal Character Encoding Detector, is needed for the proper functioning of the Jupyter Notebook, which will be installed in the next step. Furthermore, `ipympl` and `ipywidgets` are used for interactive Jupyter notebooks, `opencv-python-headless` for the analysis of image data, and `wget` for downloading an additional dataset for training. The packages `numba` translates Python code into machine code, thus speeding up Python code. This is used to make the written code in the Jupyter Notebooks faster. The other package that is installed is `jupyter notebook`, which allows for writing interactive code in notebooks, especially Python code.

#### 7.2 Manual setup

##### 7.2.1 Creating & activating conda environment 

Create a new environment with Python 3.11 and assign it a new environment name, for example, `skyrmion_unet`

```
conda create --name skyrmion_unet python=3.11
```

If the following installations does not work, one could attempt to install an environment with a different version number, e.g., changing `python=3.11` to `python=3.10`.

Activate this enviorment with the same name, in this case `skyrmion_unet`

```
conda activate skyrmion_unet
```

##### 7.2.2 Installing pip packages

First, you need to install `pip`. To do this, execute:

```
conda install pip==23.3.1
```

It's possible that `pip` is already installed for the environment, in which case you will receive a message confirming this.

Depending on whether you want to run the Skyrmion U-Net on the CPU or GPU of your machine where you're installing it, execute **only one** of the following commands. If you choose to install using GPU, you must also have a GPU available on your machine. (Alternatively, after completing the setup of one environment, you can install a second environment where you then use the other command, thus having two environments: one with the CPU version and one with the GPU version.) In case the installation fails, remove the double equal sign with the version number after the package name, e.g., `tensorflow-cpu==2.16.1` -> `tensorflow-cpu`. If issues persist, it is advisable to first remove only the version number from the package `tensorflow-cpu` or `tensorflow[and-cuda]`, and only proceed to remove other package version numbers if this does not work.

###### CPU

Install the following pip packages, when you want it to run exclusively on the CPU

```
pip install tensorflow-cpu==2.16.1 albumentations==1.4.3  matplotlib==3.8.4 pandas==2.2.1 chardet==5.2.0 ipympl==0.9.3 ipywidgets==8.1.2 opencv-python-headless==4.9.0.80 wget==3.2
```

###### GPU

Instead, in the case you want to run it on the GPU, run

```
pip install tensorflow[and-cuda]==2.16.1 albumentations==1.4.3  matplotlib==3.8.4 pandas==2.2.1 chardet==5.2.0 ipympl==0.9.3 ipywidgets==8.1.2 opencv-python-headless==4.9.0.80 wget==3.2
```

##### 7.2.3 Installing conda package

Additionally, conda packages need to be installed.  To install the packages, execute the following command:

```
conda install numba==0.59.1 anaconda::notebook==7.0.8
```

In case the installation fails, remove the double equal sign with the version number after the package name, e.g., `numba==0.59.1` -> `numba`.
