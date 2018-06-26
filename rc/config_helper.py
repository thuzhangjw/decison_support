import argparse
import json
import os

__PROJ_DIR__ = os.path.dirname(__file__)
__MODEL_DIR__ = "/models/relation_cnn"
model_name = "relation_cnn"


###############################################################################
# default configs when necessary
###############################################################################
default_train_config = {
    "model_name_suffix": "",
    "train_data_file": "data/train_sample.data",
    "fixed_word_embedding_flag": False,
    "fixed_word_vector_dim": 50,
    "fixed_word_embedding_file": "data/word2vec/anzhen50.vector",
    "random_word_vector_dim": 50,
    "record_number": -1,
    "num_epoch": 500,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "num_fold": 5,
    "filter_sizes": [2, 3, 5],
    "num_filters": 70,
    "l2_reg_lambda": 0.0,
    "word_dim": 50,
    "pos_dim": 5,
    "dist1_dim": 5,
    "dist2_dim": 5,
    "type_dim": 5,
    "drop_out_prob": 0.5
}

try:
    with open('/models/relation_cnn/model/checkpoint', mode='rt') as f:
        reuse_default = f.readline().strip().split('"')[-2]
        print("reuse default %s" % reuse_default)
except (FileExistsError, FileNotFoundError):
    print("pretrained is not found")
    reuse_default = None

default_test_config = {
    "test_data_file": "data/test_output.data",
    "test_output_file": "data/test_output.json",
    "fixed_word_embedding_flag": False,
    "fixed_word_vector_dim": 50,
    "fixed_word_embedding_file": "data/word2vec/anzhen50.vector",
    "random_word_vector_dim": 50,
    "record_number": -1,
    "reuse": reuse_default
}


###############################################################################
# load the config setting
###############################################################################
def load_train_config() -> dict:
    with open(__PROJ_DIR__ + "/train_config.json", mode='rt', encoding='utf8') as f:
        train_config = json.load(f)
    print('train config json file loaded')
    return train_config


def load_test_config() -> dict:
    with open(__PROJ_DIR__ + '/test_config.json', mode='rt', encoding='utf8') as f:
        test_config = json.load(f)
    print("test config json file loaded")
    return test_config


loaded_train_config = load_train_config()
loaded_test_config = load_test_config()


###############################################################################
# parse the train config
###############################################################################
def choose_train_config(key):
    if key in loaded_train_config.keys():
        return loaded_train_config[key]
    else:
        return default_train_config[key]


train_config_parser = argparse.ArgumentParser(description='parse train control file', prog='train')
train_config_parser.add_argument("-m", "--model_name_suffix",
                                 default=choose_train_config('model_name_suffix'))
train_config_parser.add_argument("-t", "--train_data_file",
                                 default=choose_train_config('train_data_file'))
train_config_parser.add_argument("-f", "--fixed_word_embedding_flag",
                                 default=choose_train_config('fixed_word_embedding_flag'))
train_config_parser.add_argument("--fixed_word_embedding_file",
                                 default=choose_train_config('fixed_word_embedding_file'))
train_config_parser.add_argument("--fixed_word_vector_dim",
                                 default=choose_train_config('fixed_word_vector_dim'))
train_config_parser.add_argument("--random_word_vector_dim",
                                 default=choose_train_config('random_word_vector_dim'))
train_config_parser.add_argument("--record_number",
                                 default=choose_train_config('record_number'))
train_config_parser.add_argument("--num_epoch",
                                 default=choose_train_config('num_epoch'))
train_config_parser.add_argument("--batch_size",
                                 default=choose_train_config('batch_size'))
train_config_parser.add_argument("--learning_rate",
                                 default=choose_train_config('learning_rate'))
train_config_parser.add_argument("--num_fold",
                                 default=choose_train_config('num_fold'))
train_config_parser.add_argument("--filter_sizes",
                                 default=choose_train_config('filter_sizes'))
train_config_parser.add_argument("--num_filters",
                                 default=choose_train_config('num_filters'))
train_config_parser.add_argument("--l2_reg_lambda",
                                 default=choose_train_config('l2_reg_lambda'))
train_config_parser.add_argument("--word_dim",
                                 default=choose_train_config('word_dim'))
train_config_parser.add_argument("--pos_dim",
                                 default=choose_train_config('pos_dim'))
train_config_parser.add_argument("--dist1_dim",
                                 default=choose_train_config('dist1_dim'))
train_config_parser.add_argument("--dist2_dim",
                                 default=choose_train_config('dist2_dim'))
train_config_parser.add_argument("--type_dim",
                                 default=choose_train_config('type_dim'))
train_config_parser.add_argument("--drop_out_prob",
                                 default=choose_train_config('drop_out_prob'))


###############################################################################
# parse the test config
###############################################################################
def choose_test_config(key):
    if key in loaded_test_config.keys():
        return loaded_test_config[key]
    else:
        return default_test_config[key]


test_config_parser = argparse.ArgumentParser(description='parse test control file')
test_config_parser.add_argument("-t", "--test_data_file", default=choose_test_config('test_data_file'))
test_config_parser.add_argument("-f", "--fixed_word_embedding_flag",
                                default=choose_test_config('fixed_word_embedding_flag'))
test_config_parser.add_argument("--fixed_word_embedding_file",
                                default=choose_test_config('fixed_word_embedding_file'))
test_config_parser.add_argument("--fixed_word_vector_dim",
                                default=choose_test_config('fixed_word_vector_dim'))
test_config_parser.add_argument("--random_word_vector_dim",
                                default=choose_test_config('random_word_vector_dim'))
test_config_parser.add_argument("--record_number",
                                default=choose_test_config('record_number'))
test_config_parser.add_argument("-r", "--reuse",
                                default=choose_test_config('reuse'))
test_config_parser.add_argument("--test_output_file", default=choose_test_config("test_output_file"))
test_config_parser.add_argument("-w", "--worker", default=0)
test_config_parser.add_argument("-b", "--banana", default=0)
test_config_parser.add_argument("--threads", default=9)
test_config_parser.add_argument("app", default="hello world")

