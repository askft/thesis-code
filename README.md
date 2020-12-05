# thesis-code

## Setup

### Setup using Conda (Anaconda / Miniconda)

It's best to create a custom environment first:

```
conda create -n ENV_NAME
conda activate ENV_NAME
conda install python==3.7
```

This will create an empty environment and install Python 3.7 together with
the corresponding version of pip. We will then use _that_ version of pip
to install the requirements:

```
pip install -r requirements.txt
```

It's important to get this right, since BERT requires TensorFlow 1.15,
which in its turn requires Python/pip 3.7 (not 3.8).

### Understanding the pipeline

The pipeline consists of several steps, which need not all be rerun every time.

- Step 1 is to fetch and save the data: for this purpose _either_
  `downloader` or `cord_loader` is used.
  - For `downloader`, the input is a list of newline-separated PubMed IDs.
  - For `cord_loader`, the input is the `metadata.csv` file from inside the
    `.tar.gz` files in the [CORD-19 Historical Releases](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html) (this seems unavailable for early releases).
- Step 2 is `sentencer` which processes the data further for use by the models.
- Step 3 is `ner`, named-entity recognition.
- Step 4 is `re`, relationship extraction.
- Optional step: `metrics` will create metrics such as F1-score for the NER model.
- Optional step: `analysis` will analyse the NER results to find co-occurrences.

### Running the pipeline

Open the `config.json` file in the root directory and un-ignore the steps you
want to run by setting them to `false`. Then, make sure that input and output
file names align. Here's a nice little chart to help you understand (A-H are
file names).

```
(A)———[downloader]———.                                       .——[analysis]———(E)
                      |———(C)———[sentencer]———(D)———[ner]———|
(B)———[cord_loader]——'                                       '—————[re]——————(F)

(G)———[metrics]———(H)  (independent)
```

Then run the script by doing: `python main.py`

## Converting BioBERT (TensorFlow) to ONNX

First make sure to install `tf2onnx`:

```
pip install -U tf2onnx
```

Then convert your (exported) TensorFlow model:

```
python -m tf2onnx.convert --saved-model ./PATH_TO_MODEL_DIR/ --output ./OUT_PATH/model_name.onnx
```

## Creating a symlink to a model

`ln -s [absolute path to model] [path to link]`

## Download

**[BioBERT-Base fine-tuned ONNX-model with vocabulary](https://drive.google.com/drive/folders/1neThCq4MqFPd0133WDDC4MYUycE84fT7?usp=sharing)** - fine-tuned on BC5CDR-chem dataset
