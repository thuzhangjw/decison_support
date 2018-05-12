import tensorflow as tf
import numpy as np
import ds.encoding_structure_data as esd 
import ds.processing_unstructure_data as pud 
import ds.decision_support_model as dsm 
import json 


def load_model(model_path, schema_path):
    checkpoint_file = tf.train.latest_checkpoint(model_path)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(graph=graph, config=session_conf)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
    
        ds_model = {}
        ds_model['cnn_input_x'] = graph.get_operation_by_name('input_x').outputs[0]
        ds_model['cnn_rnn_input_x'] = graph.get_operation_by_name('cr_input_x').outputs[0]
        ds_model['cnn_rnn_real_sentence_num'] = graph.get_operation_by_name('real_sentence_num').outputs[0]
        ds_model['cnn_rnn_dropout_keep_prob'] = graph.get_operation_by_name('cr_dropout_keep_prob').outputs[0]
        ds_model['cnn_rnn_batch_size'] = graph.get_operation_by_name('cr_batch_size').outputs[0]
        ds_model['ncnn_input_x'] = graph.get_operation_by_name('nc_input_x').outputs[0]
        ds_model['dropout_keep_prob'] = graph.get_operation_by_name('whole_dropout_keep_prob').outputs[0]
        ds_model['scores'] = graph.get_operation_by_name('whole_final_output/whole_scores').outputs[0]

        schema = json.load(open(schema_path, 'r'))
        ds_model['cnn_max_sentence_length'] = schema['max_words_num'] 
        ds_model['cnn_rnn_max_sentence_num'] = schema['max_sentences_num']
        ds_model['cnn_rnn_max_sentence_length'] = schema['max_words_num_per_sentence']
        sess.run(tf.tables_initializer())
    
    print('\033[1;33mdecision support model has loaded\033[0m')
    return sess, ds_model 


def predict(sess, ds_model, texts, structure_data_df):
    structure_encoded = esd.encoding(structure_data_df)
    pud.text_keyword_mapping(texts)
    unstructure_df = pud.text_participle(texts)
    
    x_cnn = dsm.get_text_for_cnn(unstructure_df['disease_his'])
    x_cnn_rnn = dsm.get_text_for_cnn_rnn(unstructure_df['disease_his'])
    x_ncnn = dsm.get_numeric_matrix_for_ncnn(structure_encoded)
    batches = dsm.batch_iter(x_cnn, x_ncnn, x_cnn_rnn, None, len(texts), 1, ds_model['cnn_max_sentence_length'], ds_model['cnn_rnn_max_sentence_num'], ds_model['cnn_rnn_max_sentence_length'], False) 
    
    cnn_batch_x, cnn_rnn_batch_x, ncnn_batch_x, sentence_num_list = zip(*(batches[0]))
    cnn_batch_x = list(cnn_batch_x)
    cnn_rnn_batch_x = list(cnn_rnn_batch_x)
    ncnn_batch_x = list(ncnn_batch_x)
    sentence_num_list = list(sentence_num_list)

    feed_dict = {
            ds_model['cnn_input_x']: cnn_batch_x,
            ds_model['cnn_rnn_input_x']: cnn_rnn_batch_x,
            ds_model['cnn_rnn_real_sentence_num']: sentence_num_list,
            ds_model['cnn_rnn_dropout_keep_prob']: 1.0,
            ds_model['cnn_rnn_batch_size']: len(texts),
            ds_model['ncnn_input_x']: ncnn_batch_x,
            ds_model['dropout_keep_prob']: 1.0
            }

    scores = sess.run(ds_model['scores'], feed_dict) 
    probs = sess.run(tf.nn.softmax(scores))
    return probs

