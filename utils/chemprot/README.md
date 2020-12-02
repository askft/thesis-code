## ChemProt
ChemProt is a annotated Chemical-Protein interaction dataset.
It includes data sets for:
* Training
* Development
* Test

There are 13 groups of relations with a total of 22 labels, which can be found in the readme.pdf for each subfolder.
## Extracting relations

Extracting relations can be done by running the `extract_relations.py`-script. To choose which dataset (*train, dev or test*) to extract the relations from, tune the variables in the beginning of the code:
````python
extra = ""
p = "chemprot/" + p_train
````
Where `p_train` can be replaced by `p_dev` or `p_test`. In the latter case, also set `extra = "_gs"`.

Further, you can choose whether or not to add entity markers to the dataset by specifying `entity_markers = True` or `entity_markers = False`.
We have found that models such as BERT and SciBERT have higher accuracy using entity markers, and in the case of sentences with multiple relations, entity markers make more sense.

After running, the output will be available in processed/ with the appropriate file names.

## Building datasets
To build your *train, dev or test* run `make_datasets.py`. The partitioning of data is decided by the following code:

````python
r_train = 0.8
r_dev = 0.1
r_test = 0.1
````
Make sure it adds upp to `1.0` or the code won't run. The script also shuffles the data, so random sampling might not be needed. Notice that the partitioning is per label as well.

Furthermore, for experimental purposes, you can choose to truncate your data to the shortest label. Truncating the data means that each label have equal amounts of occurrences. You may also set a factor less than 1.0 to scale your dataset by. This can be tuned by the following code:
````python
# Format and truncate label sets to be of equal size for experimentation
train_set   = format_dataset(train_set, 1.0, False)
dev_set     = format_dataset(dev_set, 1.0, False)
test_set    = format_dataset(test_set, 1.0, False)
````
If you do not wish to truncate or scale down your data sets, then leave the parameters as above.


### Remapping
As many of the ChemProt labels do not have many occurrences, and since many of them have to semantic difference in the context of biomedical litterature, custom labels are introduced. We also introduce a custom ID (*cid*) to simplify the fine-tuning process:

````python
    '''
    Maps ChemProt relation-labels to Custom relation-labels

    cid   Custom class                  cpr     ChemProt
    -------------------------------------------------------------------------------------
    0   NOT                             10      NOT

    1   PART-OF                         1       PART-OF

    2   INTERACTOR                      2       REGULATOR
                                        2       DIRECT-REGULATOR
                                        2       INDIRECT-REGULATOR
                                        7       MODULATOR
                                        8       CO-FACTOR
                                        9       SUBSTRATE

    3   REGULATOR-POSITIVE              3       UPREGULATOR
                                        3       ACTIVATOR
                                        3       INDIRECT-UPREGULATOR
                                        5       AGONIST
                                        5       AGONIST-ACTIVATOR
                                        7       MODULATOR-ACTIVATOR

    4   REGULATOR-NEGATIVE              4       DOWNREGULATOR
                                        4       INHIBITOR
                                        4       INDIRECT-DOWNREGULATOR
                                        6       ANTAGONIST
                                        7       MODULATOR-INHIBITOR

    5   OTHER                                   all labels not included above
    '''
````

### Statistics

Statistics are taken from the data sets during parsing and output in datasets/. The statistics count each occurrence for our custom labels and by which percentage they make up the dataset. 
