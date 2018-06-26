from os import path
import pickle
import json

from config_helper import __PROJ_DIR__, __MODEL_DIR__
from util import *


def train_data_generation(args):
    train_data_file = args.train_data_file
    weflag = args.fixed_word_embedding_flag
    if weflag:
        wefile = args.fixed_word_embedding_file
        vector_dim = args.fixed_word_vector_dim
    else:
        wefile = None
        vector_dim = args.random_word_vector_dim

    print("load train data from %s" % train_data_file)
    (sent_contents, entity1_list, entity2_list,
     sent_labels, pos_tag_list) = read_train_data(train_data_file, args)

    for label in list(set(sent_labels)):
        print("label: %s have %d instances" % (label, len([l for l in sent_labels if l == label])))

    (sent_contents, entity1_list, entity2_list,
     sent_labels, pos_tag_list) = label_filter(blocks=(sent_contents,
                                                       entity1_list,
                                                       entity2_list,
                                                       sent_labels,
                                                       pos_tag_list),
                                               drop_label='other',
                                               keep_num=300)

    for label in list(set(sent_labels)):
        print("label: %s have %d instances" % (label, len([l for l in sent_labels if l == label])))

    # Featurizer
    dist1_list, dist2_list, type_list = make_distance_features(
        sent_contents, entity1_list, entity2_list)

    # padding, make the vector within the same length
    sent_contents, seq_len = make_padded_token_list(sent_contents)
    pos_tag_list, _ = make_padded_token_list(pos_tag_list)
    dist1_list, _ = make_padded_token_list(dist1_list)
    dist2_list, _ = make_padded_token_list(dist2_list)
    type_list, _ = make_padded_token_list(type_list)

    word_dict  = make_token_dict(sent_contents)
    pos_dict   = make_token_dict(pos_tag_list)
    dist1_dict = make_token_dict(dist1_list)
    dist2_dict = make_token_dict(dist2_list)
    type_dict  = make_token_dict(type_list)

    label_dict = load_relation_dict()
    word_dict_size = len(word_dict)
    pos_dict_size = len(pos_dict)
    dist1_dict_size = len(dist1_dict)
    dist2_dict_size = len(dist2_dict)
    type_dict_size = len(type_dict)
    label_dict_size = len(label_dict)

    word_block = np.array(map_token_2_id(sent_contents, word_dict))
    pos_block = np.array(map_token_2_id(pos_tag_list, pos_dict))
    dist1_block = np.array(map_token_2_id(dist1_list, dist1_dict))
    dist2_block = np.array(map_token_2_id(dist2_list, dist2_dict))
    type_block = np.array(map_token_2_id(type_list, type_dict))
    # one hot vector for the relation label
    label_vector = map_vec_2_onehot_id(sent_labels, label_dict)

    # Word Embedding
    if weflag:
        word_vectors = load_word_vectors(word_dict, wefile, vector_dim)
    else:
        word_vectors = None
        print("use randomly generated word vectors")

    data_blocks = (word_block, pos_block, dist1_block,
                   dist2_block, type_block, label_vector)
    data_dicts = (word_dict, pos_dict, dist1_dict,
                  dist2_dict, type_dict, label_dict)
    data_dict_sizes = (word_dict_size, pos_dict_size, dist1_dict_size,
                       dist2_dict_size, type_dict_size, label_dict_size)
    return seq_len, data_blocks, data_dicts, data_dict_sizes, word_vectors


def fold_generation(data_blocks, n_folds):
    all_size = data_blocks[0].shape[0]
    all_index = list(range(all_size))
    fold_size = all_size // n_folds + 1
    from random import shuffle
    shuffle(all_index)
    fold_index_slices = [slice(f*fold_size, (f+1)*fold_size)
                       for f in range(n_folds)]

    def make_atom_fold(slc):
        return [block[slc] for block in data_blocks]

    atom_blocks_list = [make_atom_fold(s) for s in fold_index_slices]

    def make_train_test_blocks(f):
        train_atom_blocks_list = []
        for i, blocks in enumerate(atom_blocks_list):
            if i != f:
                train_atom_blocks_list.append(blocks)
            else:
                test_blocks = blocks
        train_blocks = train_atom_blocks_list[0]
        for atom_blocks in train_atom_blocks_list[1:]:
            buffer = [np.vstack([train_block, atom_block])
                      for train_block, atom_block in zip(train_blocks, atom_blocks)]
            train_blocks = buffer
        return train_blocks, test_blocks

    folded_blocks = [make_train_test_blocks(f) for f in range(n_folds)]
    return folded_blocks


def test_data_generation(args):
    test_data_file = args.test_data_file
    weflag = args.fixed_word_embedding_flag
    if weflag:
        wefile = args.fixed_word_embedding_file
        vector_dim = args.fixed_word_vector_dim
    else:
        wefile = None
        vector_dim = args.random_word_vector_dim

    print("load train data from %s" % test_data_file)
    (sent_contents, entity1_list, entity2_list,
     pos_tag_list, sent_list) = read_test_data(test_data_file, args)

    # Featurizer
    dist1_list, dist2_list, type_list = make_distance_features(
        sent_contents, entity1_list, entity2_list)

    model_file_path = path.join(__PROJ_DIR__, 'model', args.reuse)
    shape_file_path = model_file_path + ".pickle"
    with open(shape_file_path, mode='rb') as f:
        seq_len, *_ = pickle.load(f)

    # padding, make the vector within the same length
    sent_contents, seq_len = make_padded_token_list(sent_contents, pad_length=seq_len)
    pos_tag_list, _ = make_padded_token_list(pos_tag_list, pad_length=seq_len)
    dist1_list, _ = make_padded_token_list(dist1_list, pad_length=seq_len)
    dist2_list, _ = make_padded_token_list(dist2_list, pad_length=seq_len)
    type_list, _ = make_padded_token_list(type_list, pad_length=seq_len)

    model_file_path = path.join(__PROJ_DIR__, 'model', args.reuse)
    dict_file_path = model_file_path.split("_")[0] + ".dict.pickle"
    with open(dict_file_path, mode='rb') as f:
        word_dict, pos_dict, dist1_dict, dist2_dict, type_dict, _ = pickle.load(f)

    word_dict_size = len(word_dict)
    pos_dict_size = len(pos_dict)
    dist1_dict_size = len(dist1_dict)
    dist2_dict_size = len(dist2_dict)
    type_dict_size = len(type_dict)

    word_block = np.array(map_token_2_id(sent_contents, word_dict))
    pos_block = np.array(map_token_2_id(pos_tag_list, pos_dict))
    dist1_block = np.array(map_token_2_id(dist1_list, dist1_dict))
    dist2_block = np.array(map_token_2_id(dist2_list, dist2_dict))
    type_block = np.array(map_token_2_id(type_list, type_dict))
    # one hot vector for the relation label

    # Word Embedding
    if weflag:
        word_vectors = load_word_vectors(word_dict, wefile, vector_dim)
    else:
        word_vectors = None
        print("use randomly generated word vectors")

    entity_pair_list = [(e1, e2) for e1, e2 in zip(entity1_list, entity2_list)]

    data_blocks = (word_block, pos_block, dist1_block,
                   dist2_block, type_block, None)
    data_dicts = (word_dict, pos_dict, dist1_dict,
                  dist2_dict, type_dict, None)
    data_dict_sizes = (word_dict_size, pos_dict_size, dist1_dict_size,
                       dist2_dict_size, type_dict_size, 0)
    return seq_len, data_blocks, data_dicts, data_dict_sizes, word_vectors, entity_pair_list, sent_list


def test_data_parse(test_data, myjieba, args):
    (sent_contents, entity1_list, entity2_list, pos_tag_list, sent_list) = parse_raw_data(test_data, myjieba)

    # Featurizer
    dist1_list, dist2_list, type_list = make_distance_features(
        sent_contents, entity1_list, entity2_list)

    model_file_path = path.join(__PROJ_DIR__, 'model', args.reuse)
    shape_file_path = model_file_path + ".pickle"
    with open(shape_file_path, mode='rb') as f:
        seq_len, *_ = pickle.load(f)

    # padding, make the vector within the same length
    sent_contents, seq_len = make_padded_token_list(sent_contents, pad_length=seq_len)
    pos_tag_list, _ = make_padded_token_list(pos_tag_list, pad_length=seq_len)
    dist1_list, _ = make_padded_token_list(dist1_list, pad_length=seq_len)
    dist2_list, _ = make_padded_token_list(dist2_list, pad_length=seq_len)
    type_list, _ = make_padded_token_list(type_list, pad_length=seq_len)

    dict_file_path = path.join('/models/relation_cnn/model', args.reuse.split("_")[0]+".dict.pickle")
    with open(dict_file_path, mode='rb') as f:
        word_dict, pos_dict, dist1_dict, dist2_dict, type_dict, _ = pickle.load(f)

    word_dict_size = len(word_dict)
    pos_dict_size = len(pos_dict)
    dist1_dict_size = len(dist1_dict)
    dist2_dict_size = len(dist2_dict)
    type_dict_size = len(type_dict)
    
    word_block = np.array(map_token_2_id(sent_contents, word_dict))
    pos_block = np.array(map_token_2_id(pos_tag_list, pos_dict))
    dist1_block = np.array(map_token_2_id(dist1_list, dist1_dict))
    dist2_block = np.array(map_token_2_id(dist2_list, dist2_dict))
    type_block = np.array(map_token_2_id(type_list, type_dict))

    entity_pair_list = [(e1, e2) for e1, e2 in zip(entity1_list, entity2_list)]

    data_blocks = (word_block, pos_block, dist1_block,
                   dist2_block, type_block, None)
    data_dicts = (word_dict, pos_dict, dist1_dict,
                  dist2_dict, type_dict, None)
    data_dict_sizes = (word_dict_size, pos_dict_size, dist1_dict_size,
                       dist2_dict_size, type_dict_size, 0)
    word_vectors = None
    return seq_len, data_blocks, data_dicts, data_dict_sizes, word_vectors, entity_pair_list, sent_list


def load_relation_dict() -> dict:
    with open(__PROJ_DIR__ + '/data/relation_label.json', mode='rt') as f:
        label_dict = json.load(f)
    return label_dict


relation_label_dict = load_relation_dict()


def eval_relation_id(name):
    return relation_label_dict[name]


def load_entity_dict() -> dict:
    with open(__PROJ_DIR__ + "/data/entity_label.json", mode='rt') as f:
        label_dict = json.load(f)
    return label_dict


entity_label_dict = load_entity_dict()


def eval_entity_id(name):
    return entity_label_dict[name]


def label_filter(blocks, drop_label, keep_num):
    (sent_contents, entity1_list, entity2_list,
     sent_labels, pos_tag_list) = blocks
    ret_blocks = [None] * len(blocks)
    indices = [i for i, x in enumerate(sent_labels) if x != drop_label]
    keep = 0
    for i, label in enumerate(sent_labels):
        if label == drop_label:
            indices.append(i)
            keep += 1
        if keep > keep_num:
            break
    from random import shuffle
    shuffle(indices)
    for i, block in enumerate(blocks):
        ret_blocks[i] = [block[i] for i in indices]
    return ret_blocks
