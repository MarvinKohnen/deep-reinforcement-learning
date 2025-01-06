# Installation
This pages describes how to download and install the software and materials needed for this course.

## Python
For this course, we need a working installation of Python. There are two options: you can download Python yourself (see below) or you can use an online environment preconfigured with a working Python installation. 

### Online access to Python
For a quick and easy start you can access the JupyterHub of the University and log in with your Credentials. All the necessary packages will be installed and no additional setup of the environment has to be performed.

### Installing Python on your own computer
If you want to install Python on your own computer, we *highly* recommend you install Python through the [Anaconda distribution](https://www.anaconda.com/products/individual). In the box below, you can find detailed installation instructions (thanks to [the Netherlands eScience Center](https://escience-academy.github.io/2020-12-07-parallel-python/)) specific to your operating system (Mac, Windows, or Linux).

```{tabbed} Mac

1. Open the [Anaconda download page](https://www.anaconda.com/products/individual#download-section) with your web browser;
2. Download the Anaconda Installer with Python 3 for macOS (you can either use the Graphical or the Command Line Installer);
3. Install Python 3 by running the Anaconda Installer using all of the defaults for installation.

For a more detailed instruction, check out the video below:

<iframe width="560" height="315" src="https://www.youtube.com/embed/TcSAln46u9U" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

```{tabbed} Windows

1. Open the [Anaconda download page](https://www.anaconda.com/products/individual#download-section) with your web browser;
2. Download the Anaconda for Windows installer with Python 3. (If you are not sure which version to choose, you probably want the 64-bit Graphical Installer Anaconda3-...-Windows-x86_64.exe);
3. Install Python 3 by running the Anaconda Installer, using all of the defaults for installation except **make sure to check Add Anaconda to my PATH environment variable**.

For a more detailed instruction, check out the video below:

<iframe width="560" height="315" src="https://www.youtube.com/embed/xxQ0mzZ8UvA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**Note**: the video tells you to use *git bash* to test your Python installation; you may ignore this. As explained below, Windows users should use the *Anaconda prompt* instead.
```

After you have installed your own Python distribution, you can check whether it is working correctly by opening a terminal (on Mac/Linux) or Anaconda Prompt (on Windows) and running the following:

```
python -c "import sys; print(sys.executable)"
```

This command should print out the location where you installed Python, e.g., `/Users/your_name/anaconda3/bin/python` (on Mac) or `C:\Users\your_name\anaconda3\bin\python` (on Windows). 

## Downloading the material
We use [Jupyter notebooks](https://jupyter.org/) for our tutorials. The materials are stored on [Github](https://zivgitlab.uni-muenster.de/jbajorat/drl_intro-python).

The resulting directory has the following structure and contents (the # symbols represent comments/information):

```
intro-to-python                # Directory root
│                        
├── LICENSE                                      
├── README.md
├── build_book
│
├── intro-to-python            # Directory with materials
│   │
│   ├── config                 #
│   ├── getting_started        #
│   ├── img                    #
│   │
│   ├── tutorials              # Jupyter notebook tutorials
│   │ 
└── requirements.txt           # Required packages

```
:::{warning}
If you work with your own Python installation, you need to install additional Python packages to make sure all material works as expected. To do so, open a terminal (Mac/Linux) or Anaconda prompt (Windows)
and navigate to the root directory of the downloaded materials (`cd path/to/downloaded/materials`) and run the following:

    pip install -r requirements.txt

Note that this is not necessary if you use JupyterHub as your Python environment!
:::