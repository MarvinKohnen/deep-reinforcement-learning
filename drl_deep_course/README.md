# Deep Learning Course 

This repository holds the resources for the Deep Learning course as part the Deep Reinforcement Learning Lecture at the University of Münster

## Overview

First week: On regression problems.

## Setting up your Conda environment

First, I would advise you to get accustomed to conda and create yourself a programming environment for conda: Conda is an open-source package management and environment management system that runs on Windows, macOS, and Linux. It allows you to quickly install, run, and update packages and their dependencies, and manage multiple environments that can use different versions of Python and other packages. Conda helps to avoid package conflicts and ensures that your projects are reproducible.

### Why Use Conda?

* Isolation: Keep dependencies required by different projects in separate environments, avoiding conflicts.
* Reproducibility: Share your environment with others to ensure they have the same setup.
* Convenience: Easily switch between environments and manage packages with simple commands.

### Setting Up Your Python Environment with Conda

1. **Install Conda:** If you haven't installed Conda yet, you can download and install it from the [official Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. **Create a New Environment:**
   Open your terminal (or Anaconda Prompt on Windows) and create a new environment with the desired Python version:
   ```bash
   conda create --name myenv python=3.10
   ```
   Replace `myenv` with your preferred environment name and `3.8` with the Python version you need.
   
3. **Activate the Environment:**
   To start using your new environment, activate it with:
   ```bash
   conda activate myenv
   ```
4. **Install Packages:**
   Within your activated environment, you can install any package using:
   ```bash
   conda install package_name
   ```
   Replace `package_name` with the package you wish to install.

5. **Deactivate the Environment:**
   When you are done, you can deactivate your environment using:
   ```bash
   conda deactivate
   ```

For more detailed instructions, you can visit the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Additional Introductory Material for Python

* As a quick entry: The tutorial given as a Stanford course (Spring 2020 edition) [cs231n](http://cs231n.stanford.edu/). This tutorial was originally written by [Justin Johnson](https://web.eecs.umich.edu/~justincj/) for cs231n and adapted as a Jupyter notebook for cs228 by [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) and [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335) and later-on adapted by Kevin Zakka. 
* A more detailed and basic [introduction to Python (for Psychologists) by Lucas Snoek](https://lukas-snoek.com/introPy/solutions/week_1/1_python_basics.html).

You can run these introductory notebooks locally or directly on the Jupyterhub of the University of Münster, following [this link to the Jupyterhub](https://jupyterhub.wwu.de/hub/user-redirect/git-pull?repo=https%3A%2F%2Fzivgitlab.uni-muenster.de%2Fschillma%2Fintro-python&amp;urlpath=lab%2Ftree%2Fintro-python%2F&amp;branch=main) which directly loads (or updates) the current repository in your workspace on the hub and you can experiment in the notebooks (for more explanation, see those notebooks).

## Further References

For further reference:

* A more extended tutorial on python: A collection of [Tutorials on the scientific Python ecosystem](https://lectures.scientific-python.org): a quick introduction to central tools and techniques. 
* The [numpy website](https://numpy.org/doc/stable/user/index.html#user) offers more on numpy.
* and Jay Alammar offers an excellent introduction in using numpy for calculations etc. on [his blog](https://jalammar.github.io/visual-numpy/).

## Authors and acknowledgment

This repository is simply based for collecting different python introductory notebooks and serving them directly on the Jupyterhub of the University of Münster. The authors are directly mentioned in the original material.

README text was written with the help of OpenAI's GPT-4o (07/11/2024).
