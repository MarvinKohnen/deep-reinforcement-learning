# The Jupyter ecosystem
In this course, we will use several tools from *project Jupyter* {cite}`kluyver2016jupyter`. This project includes several nifty tools to make programming a little easier!

## JupyterHub
One tool that we use in this course is "JupyterHub". This piece of software allows you to easily create a preconfigured Python environment on an external server &mdash; no need to install Python on your own computer anymore! You just go to the website/public IP associated with the external server and you can start programming!

## Jupyter interfaces: classic vs. JupyterLab
The Jupyter environment offers two interfaces: the *classic* interface and the more extensive *JupyterLab* interface. On our own JupyterHub instance, we enabled the classic interface by default.

::::{note}
Both the classic interface and the JupyterLab interface are shipped with the Anaconda distribution, so no need to install them manually! If you, for whatever reason, *do* need to install them, you can do so using `pip`:

```
pip install {package-name}
```

Replace `{package-name}` with `jupyter` to install the classic notebook interface and with `jupyterlab` to (additionally) install the JupyterLab interface.

Also, while on JupyterHub (including Binder) the Jupyter environment is always running, on your own computer you need to start it manually by running the following command in your terminal (or CMD prompt/Anaconda prompt/Powershell on Windows):

```
jupyter notebook
```

which opens the classic interface. To open the JupyterLab interface, run the following:

```
jupyter lab
```

These command will open a (new) tab in your default browser with the Jupyter interface. Note that, to stop the interface on your own computer, you need to stop the terminal process you started by typing Control (or Command) + C (or simply closing the terminal).
::::

Both interfaces allow you to write, edit, and run code. Note that this is not limited to Python code! By installing different [kernels](https://jupyter.readthedocs.io/en/latest/projects/kernels.html), you can run programs written in many different programming languages, including R, [Julia](https://julialang.org/), and even Matlab. In fact, the name "Jupyter" is a reference to the three core programming languages supported by Jupyter: Julia, Python, and R.

In the Jupyter environment, code can be written and executed in different ways, which can be roughly divided into "script-based" and "interactive" approaches. In the script-based approach, you write your Python code in plain-text Python files (i.e., files with the extension `.py`) and run them, as a whole, in the terminal. In the interactive approach, you can write and run code interactively in either a Python *console* (or *shell*) or in a *Jupyter notebook*. The Jupyter notebook is the most common format for interactive computing and we will use it heavily in week 1 of this course. The next sections explains Jupyter Notebooks in more detail. 

Note that, instead of using the classic interface, you can launch the JupyterLab interface by replacing the `/tree` snippet in the URL to `/lab`. To change it back to the classic interface again, click on *Help* &rarr; *Launch Classic Notebook* or change the `/lab` snippet back to `/tree` in the URL. 

