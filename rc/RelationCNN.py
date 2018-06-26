import tensorflow as tf


class RelationCNN(object):
    def __init__(self, seq_len, data_dict_sizes, embed_dims, word_vector, num_filters, filter_sizes, l2_reg_lambda=0):

        self.model_id = hash(str(seq_len)
                             + str(data_dict_sizes)
                             + str(embed_dims)
                             + str(num_filters)
                             + str(filter_sizes)
                             + str(l2_reg_lambda))

        self.shape_info = (seq_len, data_dict_sizes, embed_dims, num_filters, filter_sizes, l2_reg_lambda)

        (word_dict_size, pos_dict_size, dist1_dict_size,
         dist2_dict_size, type_dict_size, label_dict_size) = data_dict_sizes
        word_dim = embed_dims.get('word_dim', 50) if word_vector is None else word_vector.shape[0]
        pos_dim  = embed_dims.get('pos_dim', 5)
        dist1_dim = embed_dims.get('dist1_dim', 5)
        dist2_dim = embed_dims.get('dist2_dim', 5)
        type_dim = embed_dims.get('type_dim', 5)

        emb_size = word_dim + pos_dim + dist1_dim + dist2_dim + type_dim
        self.x = tf.placeholder(tf.int32, [None, seq_len], name="x")
        self.x1 = tf.placeholder(tf.int32, [None, seq_len], name="x1")
        self.x2 = tf.placeholder(tf.int32, [None, seq_len], name="x2")
        self.x3 = tf.placeholder(tf.int32, [None, seq_len], name="x3")
        self.x4 = tf.placeholder(tf.int32, [None, seq_len], name='x4')

        self.input_y = tf.placeholder(
            tf.float32, [None, label_dict_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        if word_vector is None:
            W_wemb = tf.Variable(tf.random_uniform(
                [word_dict_size, word_dim], -1.0, +1.0))
        else:
            W_wemb = tf.Variable(word_vector)
        W_p1emb = tf.Variable(tf.random_uniform([dist1_dict_size, dist1_dim], -1.0, +1.0))
        W_p2emb = tf.Variable(tf.random_uniform([dist2_dict_size, dist2_dim], -1.0, +1.0))
        W_posemb = tf.Variable(tf.random_uniform([pos_dict_size, pos_dim], -1.0, +1.0))
        W_temb = tf.Variable(tf.random_uniform([type_dict_size, type_dim], -1.0, +1.0))

        # Embedding layer
        emb = tf.nn.embedding_lookup(W_wemb, self.x)  # word embedding
        # position from first entity embedding
        emb1 = tf.nn.embedding_lookup(W_p1emb, self.x1)
        # position from second entity embedding
        emb2 = tf.nn.embedding_lookup(W_p2emb, self.x2)
        # position of pos tag embedding
        emb3 = tf.nn.embedding_lookup(W_posemb, self.x3)  # POS embedding
        # position of type embedding
        emb4 = tf.nn.embedding_lookup(W_temb, self.x4)  # POS embedding

        # X = tf.concat(2, [emb, emb1, emb2, emb3, emb4])  # shape(?, 21, 100)
        X = tf.concat([emb, emb1, emb2, emb3, emb4], 2)
        X_expanded = tf.expand_dims(X, -1)  # shape (?, 21, 100, 1)
        l2_loss = tf.constant(0.0)
        # CNN+Maxpooling Layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, emb_size, 1, num_filters]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d( X_expanded, W, strides=[1, 1, 1, 1],
                padding="VALID", name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # shape (?, 19, 1, 70)

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h,
                ksize=[1, seq_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)  # shape= (?, 1, 1, 210)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # shape =(?, 210)

        # dropout layer
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Fully connected layer
        W = tf.Variable(tf.truncated_normal(
            [num_filters_total, label_dict_size], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[label_dict_size]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")

        # prediction and loss function
        self.predictions = tf.argmax(scores, 1, name="predictions")
        self.losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=scores, labels=self.input_y)
        self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        # Accuracy
        self.correct_predictions = tf.equal(
            self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_predictions, "float"),
            name="accuracy")

    def get_model_id(self):
        return str(self.model_id)
