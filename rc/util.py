# -*- coding: utf-8 -*-
import re

import jieba.posseg as pseg
import numpy as np
from data.make_schema import make_schema_file


def pre_process(sent):
    sent = sent.lower()
    sent = re.sub(r'\d', 'dg', sent)  # 替换数字为dg
    return sent

def parse_raw_data(test_data, myjieba):
    feature_list = [f['col_name']  for f in make_schema_file()["features"]]
    
    data_col_list = [test_data[f] if isinstance(test_data[f], list) else [test_data[f]] for f in feature_list]
    data_row_list = list(map(list, zip(*data_col_list)))
    
    sent_names = []  # 1-d array
    sent_lengths = []  # 1-d array
    sent_contents = []  # 2-d array [[w1,w2,....] ...]
    entity1_list = []  # 2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
    entity2_list = []  # 2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
    pos_tag_list = []
    sentence_list = []

    for data_row in data_row_list:
        sent = data_row[feature_list.index('sentence')]
        sentence_list.append(sent)
        sent_lengths.append(len(sent.split()))
        
        e1 = data_row[feature_list.index('entity1_name')]
        e1_s = data_row[feature_list.index('entity1_start')]
        e1_index = data_row[feature_list.index('entity1_start')]
        e1_e = data_row[feature_list.index('entity1_end')]
        e1_t = data_row[feature_list.index('entity1_type')]

        e2 = data_row[feature_list.index('entity2_name')]
        e2_s = data_row[feature_list.index('entity2_start')]
        e2_index = data_row[feature_list.index('entity2_start')]
        e2_e = data_row[feature_list.index('entity2_end')]
        e2_t = data_row[feature_list.index('entity2_type')]
        # 分词添加词性和更新实体的起止位置
        sent = pre_process(sent)
        sent_list, pos_list = [], []
        words = myjieba.posseg.cut(sent)
        cursor_s, cursor_e = 0, 0
        flag_1s, flag_1e, flag_2s, flag_2e = True, True, True, True
        #  this is to define the position of the cursers
        for word, pos in words:
            cursor_e = cursor_e + len(word)
            if cursor_s <= e1_s < cursor_e and flag_1s:
                e1_s = len(sent_list)
                flag_1s = False
            if cursor_s <= e2_s < cursor_e and flag_2s:
                e2_s = len(sent_list)
                flag_2s = False
            if cursor_s <= e1_e < cursor_e and flag_1e:
                e1_e = len(sent_list)
                flag_1e = False
            if cursor_s <= e2_e < cursor_e and flag_2e:
                e2_e = len(sent_list)
                flag_2e = False
            cursor_s = cursor_e
            sent_list.append(word)
            pos_list.append(pos)
        sent_contents.append(sent_list)
        pos_tag_list.append(pos_list)

        if e1_s < e2_s:
            entity1_list.append([e1, e1_s, e1_e, e1_t, e1_index])
            entity2_list.append([e2, e2_s, e2_e, e2_t, e2_index])
        else:
            entity1_list.append([e2, e2_s, e2_e, e2_t, e2_index])
            entity2_list.append([e1, e1_s, e1_e, e1_t, e1_index])

    return sent_contents, entity1_list, entity2_list, pos_tag_list, sentence_list

    

def read_test_data(dataset_file, args):
    fp = open(dataset_file, 'r', encoding='utf8')
    text = fp.read().replace('\r', '')
    samples = text.strip().split('\n\n\n')
    samples = samples[:args.record_number]
    sent_names = []  # 1-d array
    sent_lengths = []  # 1-d array
    sent_contents = []  # 2-d array [[w1,w2,....] ...]
    entity1_list = []  # 2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
    entity2_list = []  # 2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
    pos_tag_list = []
    sentence_list = []
    for sample in samples:
        name, sent, entities, *_ = sample.strip().split('\n')
        sentence_list.append(sent)
        sent_lengths.append(len(sent.split()))
        sent_names.append(name)
        m = re.match(
            r"\(\[['\"](.*)['\"], (\d*), (\d*), '(.*)'\],"
            r" \[['\"](.*)['\"], (\d*), (\d*), '(.*)'\]\)",
            entities.strip())
        if m:
            e1 = m.group(1)
            e1_s = int(m.group(2))  # start
            e1_e = int(m.group(3))  # end
            e1_t = m.group(4)  # title

            e2 = m.group(5)
            e2_s = int(m.group(6))
            e2_e = int(m.group(7))
            e2_t = m.group(8)
            # 分词添加词性和更新实体的起止位置
            sent = pre_process(sent)
            sent_list, pos_list = [], []
            words = pseg.cut(sent)
            cursor_s, cursor_e = 0, 0
            flag_1s, flag_1e, flag_2s, flag_2e = True, True, True, True
            #  this is to define the position of the cursers
            for word, pos in words:
                cursor_e = cursor_e + len(word)
                if cursor_s <= e1_s < cursor_e and flag_1s:
                    e1_s = len(sent_list)
                    flag_1s = False
                if cursor_s <= e2_s < cursor_e and flag_2s:
                    e2_s = len(sent_list)
                    flag_2s = False
                if cursor_s <= e1_e < cursor_e and flag_1e:
                    e1_e = len(sent_list)
                    flag_1e = False
                if cursor_s <= e2_e < cursor_e and flag_2e:
                    e2_e = len(sent_list)
                    flag_2e = False
                cursor_s = cursor_e
                sent_list.append(word)
                pos_list.append(pos)
            sent_contents.append(sent_list)
            pos_tag_list.append(pos_list)

            if e1_s < e2_s:
                entity1_list.append([e1, e1_s, e1_e, e1_t])
                entity2_list.append([e2, e2_s, e2_e, e2_t])
            else:
                entity1_list.append([e2, e2_s, e2_e, e2_t])
                entity2_list.append([e1, e1_s, e1_e, e1_t])
        # print e1,e2
        else:
            print("Error in reading", entities.strip())
            exit(0)
        fp.close()
    return sent_contents, entity1_list, entity2_list, pos_tag_list, sentence_list


def read_train_data(dataset_file, args):
    fp = open(dataset_file, 'r', encoding='utf8')
    text = fp.read().replace('\r', '')
    samples = text.strip().split('\n\n\n')
    samples = samples[:args.record_number]
    sent_names = []  # 1-d array
    sent_lengths = []  # 1-d arraypy
    sent_contents = []  # 2-d array [[w1,w2,....] ...]
    sent_lables = []  # 1-d array
    entity1_list = []  # 2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
    entity2_list = []  # 2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
    pos_tag_list = []
    for sample in samples:
        name, sent, entities, relation = sample.strip().split('\n')
        sent_lengths.append(len(sent.split()))
        sent_names.append(name)
        m = re.match(
            r"\(\[['\"](.*)['\"], (\d*), (\d*), '(.*)'\],"
            r" \[['\"](.*)['\"], (\d*), (\d*), '(.*)'\]\)",
            entities.strip())
        if m:
            e1 = m.group(1)
            e1_s = int(m.group(2))  # start
            e1_e = int(m.group(3))  # end
            e1_t = m.group(4)  # title

            e2 = m.group(5)
            e2_s = int(m.group(6))
            e2_e = int(m.group(7))
            e2_t = m.group(8)
            # 分词添加词性和更新实体的起止位置
            sent = pre_process(sent)
            sent_list, pos_list = [], []
            words = pseg.cut(sent)
            cursor_s, cursor_e = 0, 0
            flag_1s, flag_1e, flag_2s, flag_2e = True, True, True, True
            #  this is to define the position of the cursers
            for word, pos in words:
                cursor_e = cursor_e + len(word)
                if cursor_s <= e1_s < cursor_e and flag_1s:
                    e1_s = len(sent_list)
                    flag_1s = False
                if cursor_s <= e2_s < cursor_e and flag_2s:
                    e2_s = len(sent_list)
                    flag_2s = False
                if cursor_s <= e1_e < cursor_e and flag_1e:
                    e1_e = len(sent_list)
                    flag_1e = False
                if cursor_s <= e2_e < cursor_e and flag_2e:
                    e2_e = len(sent_list)
                    flag_2e = False
                cursor_s = cursor_e
                sent_list.append(word)
                pos_list.append(pos)
            sent_contents.append(sent_list)
            pos_tag_list.append(pos_list)

            if e1_s < e2_s:
                entity1_list.append([e1, e1_s, e1_e, e1_t])
                entity2_list.append([e2, e2_s, e2_e, e2_t])
            else:
                entity1_list.append([e2, e2_s, e2_e, e2_t])
                entity2_list.append([e1, e1_s, e1_e, e1_t])
        # print e1,e2
        else:
            print("Error in reading", entities.strip())
            exit(0)
        ma = re.match(
            r"\[['\"](.*)['\"], '(.*)', ['\"](.*)['\"]\]",
            relation.strip())
        label = None
        if ma:
            label = ma.group(2)
        elif relation == '[0]':
            label = 'other'
        else:
            print("Error in reading", relation)
            exit(0)
        # print lable
        sent_lables.append(label)
        fp.close()
    return sent_contents, entity1_list, entity2_list, sent_lables, pos_tag_list


def map_token_2_id(sent_contents, word_dict: dict):
    T = []
    for sent in sent_contents:
        t = []
        for w in sent:
            t.append(word_dict.get(w, 0))
        T.append(t)
    return T


def make_distance_features(sent_contents, entity1_list, entity2_list):
    d1_list = []
    d2_list = []
    type_list = []
    for sent, e1_part, e2_part in zip(
            sent_contents, entity1_list, entity2_list):
        _, s1, e1, t1, *_ = e1_part # name, start, end, type
        _, s2, e2, t2, *_ = e2_part
        maxl = len(sent)

        d1 = []
        for i in range(maxl):
            if i < s1:
                d1.append(str(i - s1))
            elif i > e1:
                d1.append(str(i - e1))
            else:
                d1.append('0')
        d1_list.append(d1)

        d2 = []
        for i in range(maxl):
            if i < s2:
                d2.append(str(i - s2))
            elif i > s2:
                d2.append(str(i - s2))
            else:
                d2.append('0')
        d2_list.append(d2)

        t = ['Out'] * maxl
        for i in range(s1, e1 + 1):
            t[i] = t1
        for i in range(s2, e2 + 1):
            t[i] = t2
        type_list.append(t)
    return d1_list, d2_list, type_list


def make_token_dict(sent_list):
    from itertools import chain
    token_unique_list = list(set(chain.from_iterable(sent_list)))
    token_dict = {w: i+1 for i, w in enumerate(token_unique_list)}
    token_dict['Unknown'] = 0
    return token_dict


def make_padded_token_list(sent_cont, pad_symbol='<pad>', pad_length=None):
    maxl = max([len(sent) for sent in sent_cont])
    maxl = maxl if pad_length is None else pad_length
    T = []
    for sent in sent_cont:
        t = []
        lenth = len(sent)
        for i in range(lenth):
            t.append(sent[i])
        for i in range(lenth, maxl):
            t.append(pad_symbol)
        T.append(t)

    return T, maxl


def map_vec_2_onehot_id(sent_lables, label_dict):
    Y_t = [label_dict[label] for label in sent_lables]
    label_dict_size = len(label_dict)
    Y_train = np.zeros((len(Y_t), label_dict_size))
    Y_train[range(len(Y_t)), Y_t] = 1
    return Y_train


def load_word_vectors(word_dict, fname, embSize=50):
    print("Reading word vectors")
    wv = []
    wl = []
    with open(fname, 'r', encoding='utf8') as f:
        for line in f:
            vs = line.split()
            if len(vs) < embSize:
                continue
            vect = list(map(float, vs[1:]))
            assert len(vect) == embSize
            wv.append(vect)
            wl.append(vs[0])
    wordemb = []
    count = 0
    for word, index in word_dict.items():
        if word in wl:
            wordemb.append(wv[wl.index(word)])
        else:
            count += 1
            wordemb.append(np.random.rand(embSize))
    wordemb = np.asarray(wordemb, dtype='float32')
    print("number of unknown word in word embedding", count)
    return wordemb


if __name__ == '__main__':
    ftrain = 'data/train_sample.data'
    (sent_contents, entity1_list, entity2_list,
     sent_lables, pos_tag_list) = read_train_data(ftrain)
