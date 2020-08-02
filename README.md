# thesis-code

## Setup

### Using Conda (Anaconda / Miniconda)

```
conda create -n ENV_NAME
conda activate ENV_NAME
conda install python==3.7
```

This will create an empty environment and install Python 3.7 together with the corresponding version of `pip`. We will then use _that_ version of `pip` to install the requirements.

```
pip install -r requirements.txt
```

It's important to get this right, since BERT requires TensorFlow 1.15, which in its turn requires Python/PIP 3.7 (not 3.8!).
