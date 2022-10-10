# MOZER : a Model OptimiZER utility

MOZER is a Python package that provides various features:

- Automatically partition deep neural networks model.

- Add quantization layer to sliced layer

- To be added..

## Python Package Installation

### MOZER package

Set the environment variable PYTHONPATH to tell python where to find the library. For example, assume we cloned tvm on the directory /path/to/mozer then we can add the following line in ~/.bashrc. The changes will be immediately reflected once you pull the code and rebuild the project (no need to call setup again)


```bash
export MOZER_HOME=/path/to/mozer
export PYTHONPATH=$MOZER_HOME/python:${PYTHONPATH}
```