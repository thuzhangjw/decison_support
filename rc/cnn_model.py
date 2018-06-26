import pickle
import os.path as path

import numpy as np
import sklearn as sk
import tensorflow as tf

from config_helper import __PROJ_DIR__, __MODEL_DIR__
from RelationCNN import RelationCNN


class cnn_model(object):
    def __init__(self, seq_len, data_dict_sizes, word_vector, model_name, args, data_dicts=None, reuse=None):
        # print 'd1_dict_size', d1_dict_size
        # print 'd2_dict_size', d2_dict_size
        # print "pos dict size", pos_dict_size
        self.sess = tf.Session()
        self.args = args
        self.word_vector = word_vector

        if reuse is not None:
            model_file_path = path.join(__PROJ_DIR__, 'model', args.reuse)
            shape_file_path = model_file_path + ".pickle"
            with open(shape_file_path, mode='rb') as f:
                (seq_len_, data_dict_sizes_, embed_dims_,
                 num_filters_, filter_sizes_, l2_reg_lambda_) = pickle.load(f)
            self.cnn = RelationCNN(
                seq_len=seq_len_,
                data_dict_sizes=data_dict_sizes_,
                word_vector=self.word_vector,
                embed_dims=embed_dims_,
                filter_sizes=filter_sizes_,
                num_filters=num_filters_,
                l2_reg_lambda=l2_reg_lambda_)
            self.model_name = reuse
            self.save_path = path.join(__PROJ_DIR__, 'model', self.model_name)
        else:
            self.cnn = RelationCNN(
                seq_len=seq_len,
                data_dict_sizes=data_dict_sizes,
                word_vector=self.word_vector,
                embed_dims={"word_dim": args.word_dim,
                            "pos_dim": args.pos_dim,
                            "dist1_dim": args.dist1_dim,
                            "dist2_dim": args.dist2_dim,
                            "type_dim": args.type_dim},
                filter_sizes=args.filter_sizes,
                num_filters=args.num_filters,
                l2_reg_lambda=args.l2_reg_lambda)
            self.model_name = model_name + self.cnn.get_model_id()
            self.optimizer = tf.train.AdamOptimizer(args.learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.cnn.loss)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars, global_step=self.global_step)
            self.sess.run(tf.initialize_all_variables())
            self.save_path = path.join(__PROJ_DIR__, 'model', self.model_name)
            dict_file_path = self.save_path + ".dict.pickle"
            with open(dict_file_path, mode='wb') as f:
                pickle.dump(data_dicts, f)


        self.saver = tf.train.Saver()
        if reuse is not None:
            self.restore()

        # self.fp = open("result.txt",'a')

    def train_step(self, W_batch, d1_batch, d2_batch, P_batch, T_batch, y_batch, fold, save_every):
        feed_dict = {
            self.cnn.x: W_batch,
            self.cnn.x1: d1_batch,
            self.cnn.x2: d2_batch,
            self.cnn.x3: P_batch,
            self.cnn.x4: T_batch,
            self.cnn.input_y: y_batch,
            self.cnn.dropout_keep_prob: self.args.drop_out_prob
        }
        _, step, loss, accuracy, predictions = self.sess.run(
            [self.train_op,
             self.global_step,
             self.cnn.loss,
             self.cnn.accuracy,
             self.cnn.predictions],
            feed_dict)
        if step % save_every == 0:
            self.store("%s_fold_%s_step_%s" % (self.save_path, str(fold), str(step)))
            print("step " + str(step) + " loss " + str(loss) + " accuracy " + str(accuracy))

    def set_args(self, args):
        self.args = args

    def test_step(self, W_batch, d1_batch, d2_batch, P_batch, T_batch, y_batch):
        feed_dict = {
            self.cnn.x: W_batch,
            self.cnn.x1: d1_batch,
            self.cnn.x2: d2_batch,
            self.cnn.x3: P_batch,
            self.cnn.x4: T_batch,
            self.cnn.input_y: y_batch,
            self.cnn.dropout_keep_prob: 1.0
        }
        step, loss, accuracy, predictions = self.sess.run(
            [self.global_step,
             self.cnn.loss,
             self.cnn.accuracy,
             self.cnn.predictions],
            feed_dict)
        print("Accuracy in test data", accuracy)
        return accuracy, predictions

    def cnn_train(self, train_blocks, test_blocks, fold, save_every=100):
        (W_tr, P_tr, d1_tr, d2_tr, T_tr, Y_tr) = train_blocks
        (W_te, P_te, d1_te, d2_te, T_te, Y_te) = test_blocks
        # print P_tr
        batch_size = self.args.batch_size
        for _ in range(self.args.num_epoch):
            end = 1
            j = 0
            while end > 0:
                begin, end = j * batch_size, (j+1) * batch_size
                if end >= len(W_tr):
                    end = -1
                s = slice(begin, end)
                self.train_step(W_tr[s], d1_tr[s], d2_tr[s],
                                P_tr[s], T_tr[s], Y_tr[s],
                                fold, save_every=save_every)
                j += 1

        acc, pred = self.test_step(W_te, d1_te, d2_te, P_te, T_te, Y_te)
        y_true = np.argmax(Y_te, 1)
        y_pred = pred
        f1_score = sk.metrics.f1_score(
            y_true, y_pred, pos_label=None, average='weighted')
        return acc, f1_score

    def predict(self, input_blocks):
        feed_dict = {
            self.cnn.x: input_blocks[0],
            self.cnn.x1: input_blocks[2],
            self.cnn.x2: input_blocks[3],
            self.cnn.x3: input_blocks[1],
            self.cnn.x4: input_blocks[4],
            self.cnn.dropout_keep_prob: 1.0
        }
        y = self.sess.run(self.cnn.predictions, feed_dict)
        return y

    def store(self, model_file_path):
        print("storing the model info within %s" % model_file_path)
        self.saver.save(self.sess, model_file_path)
        shape_file_path = model_file_path+".pickle"
        with open(shape_file_path, mode='wb') as f:
            pickle.dump(self.cnn.shape_info, f)
        print("Done")

    def restore(self):
        self.saver.restore(self.sess, self.save_path)
