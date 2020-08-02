# thesis-code

## Setup

### Using Conda (Anaconda / Miniconda)

```
conda create -n ENV_NAME
conda activate ENV_NAME
conda install python==3.7
```

This will create an empty environment and install Python 3.7 together with
the corresponding version of `pip`. We will then use _that_ version of `pip`
to install the requirements.

```
pip install -r requirements.txt
```

It's important to get this right, since BERT requires TensorFlow 1.15,
which in its turn requires Python/PIP 3.7 (not 3.8!).


## Converting BioBERT (TensorFlow) to ONNX
First make sure to install tf2onnx
```
pip install -U tf2onnx
```
Then convert your (exported) TensorFlow-model.
```
python -m tf2onnx.convert --saved-model ./PATH_TO_MODEL_DIR/ --output ./OUT_PATH/model_name.onnx
```
