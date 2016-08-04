import tensorflow as tf
from constructors.Layer import Layer
# from ...logdir import logdir
import time

class FCN(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.layers = []
        self.targs = tf.placeholder(tf.float32, shape=(None, sizes[-1]), name='Output_placeholder')
        _first = True
        for depth, num_units in enumerate(sizes[:-1]):
            if _first:
                # Initialize input layer
                self.layers.append(Layer(depth = 0,
                                         dims = (num_units, None),
                                         type = 'input',
                                         sender = None,
                                         input_layer=True))
                _first = False
            else:
                # Initialize hidden layers
                self.layers.append(Layer(depth = depth,
                                         dims = (num_units, self.sizes[depth-1]),
                                         type = 'hidden',
                                         sender = self.layers[-1]))
        # Initialize output layer
        self.layers.append(Layer(depth = self.num_layers-1,
                                 dims = (self.sizes[-1],self.sizes[-2]),
                                 type = 'output',
                                 sender = self.layers[-1]))
        self.loss = tf.reduce_sum(tf.squared_difference(self.targs, self.layers[-1].activation) / 2, name='squared_error')

    def describe(self):
        for i, layer in enumerate(self.layers):
            print('=' * 50, 'Layer_{}'.format(layer.name), '=' * 50)
            print('| Activation function:   ', layer.act_func)
            print('| Number of units:       ', layer.size)
            if i > 0:
                print('| Weights:', ' ' * 14, layer.W)
                print('| Biases:' + ' ' * 16, layer.b)
                print('| Input tensor:' + ' ' * 10, layer.net_inp, ',', layer.net_inp.name)
            print('| Output tensor:' + ' ' * 9, layer.activation, ',', layer.activation.name)
            print('=' * 116)

    def zero_all_weights(self):
        for layer in self.layers[1:]:
            layer.weights_to_zeros()

    def tune_all_weights(self, wrange, constraint='np'):
        for layer in self.layers[1:]:
            layer.weights_to_range(wrange=wrange, constraint=constraint)

    def f_prop(self, x):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            y = sess.run(self.layers[-1].activation, feed_dict={self.layers[0].activation: x})
            return y

    def define_loss(self, loss_func='squared_error'):
        y_ = self.targs
        y = self.layers[-1].activation
        if loss_func=='cross_entropy':
            def clipped(x):
                # this handles cases when y * tf.log(y') outputs NaN
                return tf.clip_by_value(x, 1e-10, 1.0)
            self.loss = -tf.reduce_sum(y_ * tf.log(clipped(y)) + (1 - y_) * tf.log(clipped(1 - y)), name='cross_entropy')
        elif loss_func=='squared_error':
            self.loss = tf.reduce_sum(tf.squared_difference(y_, y) / 2, name='squared_error')

    def train(self, train_data, numepochs=30, batch_size=4, lrate=0.5, momentum=0.9):
        opt = tf.train.MomentumOptimizer(lrate, momentum)
        dEds = opt.compute_gradients(self.loss)
        #for pair in dEds: print(pair)
        learn = opt.apply_gradients(dEds)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            start_time = time.time()
            i = 0
            while i < numepochs:
                batch_xs, batch_ys = train_data.next_batch(batch_size)
                feed = {self.layers[0].activation: batch_xs, self.targs: batch_ys}
                sess.run(learn, feed_dict=feed)
                i += 1
                if i == numepochs:
                    print('Training complete!\n')
                    end_time = time.time()
                    print('Training took about {} second(s):'.format(round(end_time - start_time)))
                print('Final error = {}'.format(sess.run(self.loss, feed_dict=feed)))


    def fill_feed_dict(self, data_set, images_pl, labels_pl, batch_size):
        """Fills the feed_dict for training the given step.
        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }
        Args:
          data_set: The set of images and labels, from input_data.read_data_sets()
          images_pl: The images placeholder, from placeholder_inputs().
          labels_pl: The labels placeholder, from placeholder_inputs().
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        # Create the feed_dict for the placeholders filled with the next
        # `batch size ` examples.
        images_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            images_pl: images_feed,
            labels_pl: labels_feed,
        }
        return feed_dict

    def summarize(self):
        merged = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(os.getcwd() + 'logdir/events', graph=tf.get_default_graph())
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

    # def inference(self, hidden1_units, hidden2_units):
    #     """Build the MNIST model up to where it may be used for inference.
    #     Args:
    #       images: Images placeholder, from inputs().
    #       hidden1_units: Size of the first hidden layer.
    #       hidden2_units: Size of the second hidden layer.
    #     Returns:
    #       softmax_linear: Output tensor with the computed logits.
    #     """
    #     # Hidden 1
    #     with tf.name_scope('hidden1'):
    #         weights = tf.Variable(
    #             tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
    #                                 stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    #             name='weights')
    #         biases = tf.Variable(tf.zeros([hidden1_units]),
    #                              name='biases')
    #         hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    #     # Hidden 2
    #     with tf.name_scope('hidden2'):
    #         weights = tf.Variable(
    #             tf.truncated_normal([hidden1_units, hidden2_units],
    #                                 stddev=1.0 / math.sqrt(float(hidden1_units))),
    #             name='weights')
    #         biases = tf.Variable(tf.zeros([hidden2_units]),
    #                              name='biases')
    #         hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    #     # Linear
    #     with tf.name_scope('softmax_linear'):
    #         weights = tf.Variable(
    #             tf.truncated_normal([hidden2_units, NUM_CLASSES],
    #                                 stddev=1.0 / math.sqrt(float(hidden2_units))),
    #             name='weights')
    #         biases = tf.Variable(tf.zeros([NUM_CLASSES]),
    #                              name='biases')
    #         logits = tf.matmul(hidden2, weights) + biases
    #     return logits

