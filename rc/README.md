# Relation Extraction IDS Module

## Todo list

1. Prepare the model store and restore mechanism [Done]
2. Fit the input output interface [Doing]
3. 

## Model description

This module is used for two purposes, training and Relation Discovering.
For the training part, this model does not need pre trained word vectors.
Meanwhile, the auto generated word vectors will be trained in our model.

We assume that for every entity pair tuple (entity 1, entity 2, sentence) there
is only one label for that. Which is more clear in the medical relation.

Interestingly, the information of word embeddings are automatically stored in
the tensorflow checkpoints.

## Training Part

### Input

Training Data in `data` folder with entity annotated in the form like sample
`data/train_sample.data`.

### config

The config file is in `train_config.json` and loaded by `config_helper.py`.
The config is in three parts:
1. `model_name_suffix`: specify the model name
2. `dataset_config`: everything that needed to know about the datasets
3. `train_config`: settings for the train optimizer
4. `RelationCNN_config`: settings for the CNN and loss

### processing pipeline

1. firstly convert the input data into the training format, especially:
    1. the iterate that stores the data `data_blocks`
    2. the dictionaries that made the conversion `data_dicts`
    3. the length information `data_shapes`

### output

1. Tensorflow checkpoints that could be reused in `model` folder, with a file
    named `model_list.txt` is maintained for registration check.
2. The model with input encoding.
3. The results for the trained model in `result` folder

## Relation Discovering Part

### Input
1. Test configuration in `test_config.json` file.
2. the input file in form `data/test_sample.json`

### output
1. The results for the predictions