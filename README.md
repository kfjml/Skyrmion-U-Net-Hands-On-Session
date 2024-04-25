# Welcome to the Skyrmion U-Net hands-on repository !

This repository is based on and is an extraction of the following Zenodo repository (v2.0) from Winkler et al.: https://doi.org/10.5281/zenodo.10997175. It is intended for getting started with the Skyrmion U-Net. The authors of this repository are Kilian Leutner, Thomas Brian Winkler, Isaac Labrie-Boulay, Alena Romanova, Hans Fangohr, Mathias Kläui, Raphael Gruber, Fabian Kammerbauer, Klaus Raab, Jakub Zazvorka, Elizabeth Martin Jefremovas and Robert Frömter.

For any questions regarding this repository, please contact Kilian Leutner at kileutne@students.uni-mainz.de.

## Installation & start 
### 1. Install & start miniconda

Install newest version of miniconda

[https://docs.anaconda.com/free/miniconda/](https://docs.anaconda.com/free/miniconda/)
[https://docs.anaconda.com/free/miniconda/miniconda-install/](https://docs.anaconda.com/free/miniconda/miniconda-install/)

and start/initialize conda

### 2. Setup of the skyrmion U-Net enviroment

#### 2.0 Preinstallations for the GPU

If you intend to run the Skyrmion U-Net on the GPU, you need to install the NVIDIA GPU driver. You can test the driver's functionality using `nvidia-smi`.

For further information, as the Skyrmion U-Net is based on TensorFlow, please refer to [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip). Only follow the steps up to `Step 3: Install TensorFlow` as the subsequent steps are explained in this README.


#### 2.1 Setup with a script

You can install both enviroments for CPU and GPU independently of each other, and also both together.

#### 2.1.1 CPU enviroment

When you want to run the skyrmion U-Net exclusively on the CPU, run the following command which will create a conda environment `skyrmion_unet_cpu` for the CPU usage with all necessary packages

```
conda env create -f environment.yml
```

#### 2.1.2 GPU enviroment

When you want to run the skyrmion U-Net exclusively on the GPU, run the following command which will create a conda environment `skyrmion_unet_gpu` for the GPU usage with all necessary packages

```
conda env create -f environment_gpu.yml
```

#### 2.2 Manual setup

#### 2.2.1 Creating & activating conda enviroment

Create a new environment with Python 3.11 and assign it a new environment name, for example, `skyrmion_unet`

```
conda create --name skyrmion_unet python=3.11
```

Activate this enviorment with the same name, in this case `skyrmion_unet`

```
conda activate skyrmion_unet
```

#### 2.2.2 Installing pip packages

Normally, when creating a new conda environment, pip is installed automatically. For confirmation, you can check this when the Skyrmion UNet environment is activated, with e.g. `which pip`. In case it is not installed, execute `conda install pip==23.3.1` for installation of pip.

First, pip packages are installed. Among these, `tensorflow` is the package used for machine learning and neural networks, and the `albumentations` package is used for data augmentation during training. The `matplotlib` package is used for plotting data, especially for MOKE images and their predictions with the Skyrmion U-Net. Furthermore, the `pandas` package is used for data saving, analysis, and editing. Lastly, the `chardet` package, which is a Universal Character Encoding Detector, is needed for the proper functioning of the Jupyter Notebook, which will be installed in the next step. 

Depending on whether you want to run the Skyrmion U-Net on the CPU or GPU of your machine where you're installing it, execute **only one** of the following commands. If you choose to install using GPU, you must also have a GPU available on your machine. (Alternatively, after completing the setup of one environment, you can install a second environment where you then use the other command, thus having two environments: one with the CPU version and one with the GPU version.)

##### CPU

Install the following pip packages, when you want it to run exclusively on the CPU

```
pip install tensorflow-cpu==2.16.1 albumentations==1.4.3  matplotlib==3.8.4 pandas==2.2.1 chardet==5.2.0 ipympl==0.9.3 ipywidgets==8.1.2 opencv-python-headless==4.9.0.80 wget==3.2
```

##### GPU

Instead, in the case you want to run it on the GPU, run

```
pip install tensorflow[and-cuda]==2.16.1 albumentations==1.4.3  matplotlib==3.8.4 pandas==2.2.1 chardet==5.2.0 ipympl==0.9.3 ipywidgets==8.1.2 opencv-python-headless==4.9.0.80 wget==3.2
```

#### 2.2.3 Installing conda packages

Additionally, conda packages need to be installed. One of these packages is `numba`, which translates Python code into machine code, thus speeding up Python code. This is used to make the written code in the Jupyter Notebooks faster. The other package that is installed is `jupyter notebook`, which allows for writing interactive code in notebooks, especially Python code. To install the packages, execute the following command:

```
conda install numba==0.59.1 anaconda::notebook==7.0.8
```

### 3 Using the skyrmion U-Net

The script/manual guide automatically activates the Skyrmion U-Net Conda environment. To activate the environment again later, you use the environment name. In the case that you installed the CPU environment, the name is `skyrmion_unet_cpu`; in the case that you installed the GPU environment, the name is `skyrmion_unet_gpu`; or in the case of the manual installation, use the chosen enviroment name (the standard was `skyrmion_unet`). With this name, activate the environment using the following command and the associated environment name, in the following command `skyrmion_unet`:

```
conda activate skyrmion_unet
```

Afterwards start a jupyter notebook with

```
jupyter notebook
```

This command will provide you with a localhost address, which you can then open in your web browser to access the Jupyter Notebook. Then you can try out the Jupyter notebooks for training (`Training.ipynb`) and prediction (`Prediction.ipynb`) in the main folder of the repository, and also create your own code for the Skyrmion U-Net.