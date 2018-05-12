import tensorflow as tf
import pandas as pd
from ds.cnn import CNNLayer
from ds.numeric_cnn import NumericCNN
from ds.cnn_rnn import CNN_RNN
import ds.common 
import re
import random 
import os 
import datetime 
import numpy as np 


def generate_feature_dim_list(column_names):
    feature_dim_list = []
    current_name = column_names[0].split('_')[0]
    l = 0
    for name in column_names:
        tn = name.split('_')[0]
        if tn == current_name:
            l += 1
        else:
            feature_dim_list.append(l)
            l = 1
            current_name = tn
    feature_dim_list.append(l)
    assert(len(column_names) == sum(feature_dim_list))
    return feature_dim_list


def reshape2matrix(line_feature, feature_dim_list, max_feature_dim):
    i = 0
    res = []
    for l in feature_dim_list:
        tmp = [0.0] * i
        tmp += line_feature[i:i+l]
        tmp += [0.0] * (max_feature_dim - (i+l))
        res.append(tmp)
        i += l
    assert(i == len(line_feature))
    return res


def convert_word2_onehot(word_labels):
    labels =[]
    for dis in word_labels:
        label = [0] * len(ds.common.CLASSES)
        label[ds.common.CLASSES.index(dis)] = 1
        labels.append(label)

    return labels 


def get_text_for_cnn(texts):
    return [ re.sub(r'。', '', text) for text in texts ]


def get_text_for_cnn_rnn(texts):
    tmp = [text.split('。') for text in texts]
    return [ [text, text[:-1]][text[-1]=='']  for text in tmp ]
 

def get_numeric_matrix_for_ncnn(structure_df):
    feature_dim_list = generate_feature_dim_list(structure_df.columns)
    max_feature_dim = sum(feature_dim_list)
    numeric_features = structure_df.astype('float32').values.tolist()
    numeric_matrix_features = list(map(lambda x: reshape2matrix(x, feature_dim_list, max_feature_dim), numeric_features))
    return numeric_matrix_features


def split_train_test_set(df):
    groups = df.groupby(ds.common.LABEL_NAME).groups
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    criterion = [True] * len(df)
    for group in sorted_groups:
        num = 0
        l = len(group[1])
        while num < 0.1 * l:
            i = random.randint(0, l-1)
            if criterion[group[1][i]]:
                num += 1
                criterion[group[1][i]] = False

    train = df[criterion].reset_index(drop=True)
    test = df[list(map(lambda x: not x, criterion))].reset_index(drop=True)
    return train, test 


def get_cnn_rnn_max_sentence_length(x_set):
    res = 0
    for doc in x_set:
        for sentence in doc:
            sl = len(sentence.strip().split(' '))
            if sl > res:
                res = sl
    return res 


def shuffle_and_batch(data, batch_size, epochs, shuffle=True):
    s = pd.Series(list(data))
    data_size = len(s)
    batch_nums = int((len(s) - 1) / batch_size) + 1
    res = []
    for _ in range(epochs):
        if shuffle:
            shuffled_data = s.sample(frac=1)
        else:
            shuffled_data = s
        for batch_num in range(batch_nums):
            start_index = batch_num * batch_size
            end_index = min(data_size, (batch_num + 1)* batch_size)
            res.append(list(shuffled_data[start_index:end_index]))
    return res 


def batch_iter(x_cnn, x_ncnn, x_cnn_rnn, y, batch_size, epochs, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length, shuffle=True):

    def helper(words, max_length):
        if len(words) > max_length:
            return words[0:max_length]
        else:
            return words+['unknown']*(max_length-len(words))

    padded_x_cnn = list(map(lambda a: helper(a.strip().split(' '), cnn_max_sentence_length), x_cnn))
    #if len(x) > cnn_max_sentence_length:
    #    padded_x_cnn = x[0:cnn_max_sentence_length]
    #else:
    #    padded_x_cnn = list(map(lambda a: a+['unknown']*(cnn_max_sentence_length - len(a)), x))
 
    sentence_num_list = [len(a) for a in x_cnn_rnn]
    padded_x_cnn_rnn = []
    for doc in x_cnn_rnn:
        tmp = []
        for sentence in doc:
            ns = sentence.strip().split(' ')
            if len(ns) > cnn_rnn_max_sentence_length:
                ns = ns[0:cnn_rnn_max_sentence_length]
            else:
                ns += ['unknown'] * (cnn_rnn_max_sentence_length - len(ns))
            tmp.append(ns)
        if len(tmp) > cnn_rnn_max_sentence_num:
            tmp = tmp[0:cnn_rnn_max_sentence_num]
        else:
            for _ in range(cnn_rnn_max_sentence_num - len(doc)):
                tmp.append(['unknown']*cnn_rnn_max_sentence_length)
        padded_x_cnn_rnn.append(tmp) 

    if y == None:
        data = zip(padded_x_cnn, padded_x_cnn_rnn, x_ncnn, sentence_num_list)
    else:
        data = zip(padded_x_cnn, padded_x_cnn_rnn, x_ncnn, sentence_num_list, y)
    return shuffle_and_batch(data, batch_size, epochs, shuffle)


def train(df, model_name, word2vec_model, model_save_dir, batch_size=200, epochs=150, cnn_filter_sizes=[2,3,4,5], cnn_num_filters=100, cnn_l2_reg_lambda=0.5, rnn_hidden_dim=300, rnn_l2_reg_lambda=0.25, ncnn_filter_sizes=[2,3,4,5,7,8,11,13,19,23,25], ncnn_num_filters=50, ncnn_l2_reg_lambda=0.2, whole_l2_reg_lambda=0.1, rnn_dropout_keep_prob=0.6, whole_dropout_keep_prob=0.4):
    
    train_df, test_df = split_train_test_set(df)

    y_train = convert_word2_onehot(train_df[ds.common.LABEL_NAME])
    x_train_cnn = get_text_for_cnn(train_df['disease_his'])
    x_train_cnn_rnn = get_text_for_cnn_rnn(train_df['disease_his'])
    x_train_ncnn = get_numeric_matrix_for_ncnn(train_df.iloc[:, 2:])

    y_test = convert_word2_onehot(test_df[ds.common.LABEL_NAME])
    x_test_cnn = get_text_for_cnn(test_df['disease_his'])
    x_test_cnn_rnn = get_text_for_cnn_rnn(test_df['disease_his'])
    x_test_ncnn  = get_numeric_matrix_for_ncnn(test_df.iloc[:, 2:])

    cnn_max_sentence_length = max([len(x.split(' ')) for x in (x_train_cnn + x_test_cnn)])
    ncnn_feature_num = len(x_train_ncnn[0])
    ncnn_feature_dim = len(x_train_ncnn[0][0])
    
    cnn_rnn_max_sentence_num = max([len(a) for a in (x_train_cnn_rnn + x_test_cnn_rnn)])
    cnn_rnn_max_sentence_length = get_cnn_rnn_max_sentence_length(x_train_cnn_rnn + x_test_cnn_rnn)
    init_words_embedded_model = word2vec_model 
    num_classes = len(y_train[0])
 
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True 
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            input_y = tf.placeholder(tf.float32, [None, num_classes], name='final_input_y')
            dropout_keep_prob = tf.placeholder(tf.float32, [], name='whole_dropout_keep_prob')
            cnn = CNNLayer(
                    sequence_length=cnn_max_sentence_length,
                    filter_sizes=cnn_filter_sizes,
                    num_filters=cnn_num_filters,
                    init_words_embedded_model=init_words_embedded_model,
                    num_classes=num_classes,
                    l2_reg_lambda=cnn_l2_reg_lambda,
                    use_static=True
                    )
            

            cnn_rnn = CNN_RNN(
                    sentence_num=cnn_rnn_max_sentence_num,
                    sentence_length=cnn_rnn_max_sentence_length,
                    filter_sizes=cnn_filter_sizes,
                    num_filters=cnn_num_filters,
                    init_words_embedded_model=init_words_embedded_model,
                    rnn_hidden_dim=rnn_hidden_dim,
                    num_classes=num_classes,
                    l2_reg_lambda=rnn_l2_reg_lambda,
                    use_static=True 
                    )
            ncnn = NumericCNN(
                    feature_num=ncnn_feature_num,
                    feature_dim=ncnn_feature_dim,
                    filter_sizes=ncnn_filter_sizes,
                    num_filters=ncnn_num_filters,
                    num_classes=num_classes,
                    l2_reg_lambda=ncnn_l2_reg_lambda
                    )
 
            merged_feature = tf.concat([cnn.final_feature, cnn_rnn.final_feature, ncnn.final_feature], 1)
            l2_loss = cnn.l2_loss + cnn_rnn.l2_loss + ncnn.l2_loss
            l2_reg_lambda = whole_l2_reg_lambda 

            with tf.variable_scope('whole_final_output'):
                h_drop = tf.nn.dropout(merged_feature, dropout_keep_prob)
                W = tf.get_variable(
                        'W',
                        shape=[merged_feature.shape[1].value, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer()
                        )
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
                l2_loss += l2_reg_lambda * tf.nn.l2_loss(W) + l2_reg_lambda * tf.nn.l2_loss(b)
                whole_scores = tf.nn.xw_plus_b(h_drop, W, b, name='whole_scores')
                whole_predictions = tf.argmax(whole_scores, 1, name='whole_predictions')

            with tf.variable_scope('whole_final_loss'):
                entropy_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=whole_scores, labels=input_y)
                whole_loss = tf.reduce_mean(entropy_losses) + l2_loss

            with tf.variable_scope('whole_final_accuracy'):
                correct_predictions = tf.equal(whole_predictions, tf.argmax(input_y, 1))
                whole_accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(whole_loss, global_step=global_step)

            out_dir = os.path.join(model_save_dir, model_name) 
            print('Writing to {}\n'.format(out_dir))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, model_name)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
  
            def test(batches):
                print('Evaluate')
                all_predictions = []
                for batch in batches:  
                    cnn_batch_x, cnn_rnn_batch_x, ncnn_batch_x, sentence_num_list, y = zip(*batch)
                    cnn_batch_x = list(cnn_batch_x)
                    cnn_rnn_batch_x = list(cnn_rnn_batch_x)
                    ncnn_batch_x = list(ncnn_batch_x)
                    sentence_num_list = list(sentence_num_list)
                    y = list(y)
                    feed_dict = {
                            cnn.input_x: cnn_batch_x,
                            cnn_rnn.input_x: cnn_rnn_batch_x,
                            cnn_rnn.real_sentence_num: sentence_num_list,
                            cnn_rnn.dropout_keep_prob: 1.0,
                            cnn_rnn.batch_size: len(y),
                            ncnn.input_x: ncnn_batch_x,
                            dropout_keep_prob: 1.0
                            }

                    batch_predictions = sess.run(whole_predictions, feed_dict)
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

                y_real = np.array(list(map(lambda a: a.index(1), y_test)))
                correct_predictions = sum(all_predictions == y_real)
                print("Total number of test examples: {}".format(len(y_real)))
                print("Accuracy: {:g}".format(correct_predictions/len(y_real)))
                return correct_predictions / len(y_real) 


            max_accuracy = 0.0
            def batch_train(batches, save_model=False):
                nonlocal max_accuracy 
                for batch in batches:
                    cnn_batch_x, cnn_rnn_batch_x, ncnn_batch_x, sentence_num_list, y = zip(*batch)
                    cnn_batch_x = list(cnn_batch_x)
                    cnn_rnn_batch_x = list(cnn_rnn_batch_x)
                    ncnn_batch_x = list(ncnn_batch_x)
                    sentence_num_list = list(sentence_num_list)
                    y = list(y)
                    feed_dict = {
                            cnn.input_x: cnn_batch_x,
                            cnn_rnn.input_x: cnn_rnn_batch_x,
                            cnn_rnn.batch_size: len(y),
                            cnn_rnn.real_sentence_num: sentence_num_list,
                            cnn_rnn.dropout_keep_prob: rnn_dropout_keep_prob,
                            ncnn.input_x: ncnn_batch_x,
                            input_y: y,
                            dropout_keep_prob: whole_dropout_keep_prob 
                            }
                    _, f_step, f_loss, f_accuracy = sess.run([train_op, global_step, whole_loss, whole_accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, f_step, f_loss, f_accuracy))
                    if not save_model:
                        current_step = tf.train.global_step(sess, global_step)
                        print(current_step)
                        if current_step % 50 == 0:
                            test_accuracy = test(test_batches)
                            if test_accuracy > max_accuracy:
                                max_accuracy = test_accuracy 

                if save_model:
                    path = saver.save(sess, checkpoint_prefix)
                    print("Saved model checkpoint to {}\n".format(path))


            train_batches = batch_iter(x_train_cnn, x_train_ncnn, x_train_cnn_rnn, y_train, batch_size, epochs, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length)
            test_batches = batch_iter(x_test_cnn, x_test_ncnn, x_test_cnn_rnn, y_test, batch_size, 1, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length, False)
            
            batch_train(train_batches)

            wholebatches = batch_iter(x_train_cnn+x_test_cnn, x_train_ncnn+x_test_ncnn, x_train_cnn_rnn+x_test_cnn_rnn, y_train+y_test, batch_size, epochs, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length)
            
            batch_train(wholebatches, save_model=True)
            print('The test accuracy is \033[1;33m%f\033[0m' % max_accuracy)
            return cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length 

