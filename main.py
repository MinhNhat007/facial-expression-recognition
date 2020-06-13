import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

from sklearn.utils import shuffle


def to_one_hot_encoded(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    ind[np.arange(y.size), y] = 1
    return ind


def init_filter(shape, pool_sz):
    w = np.random.randn(*shape) * math.sqrt(2) / math.sqrt(
        np.prod(shape[:-1]) + shape[-1] * np.prod(shape[:-2] / np.prod(pool_sz)))
    return w.astype(np.float32)


def init_weight_and_bias(input_size, output_size):
    W = np.random.randn(input_size, output_size) / np.sqrt(input_size)
    b = np.zeros(output_size)
    return W.astype(np.float32), b.astype(np.float32)


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


class DataProcessor:
    def __init__(self, file_name):
        self.X, self.Y = self.__read_data(file_name)
        self.X, self.Y = shuffle(self.X, self.Y)

    def process_data(self, n_test, choosen):
        # shuffle and split
        X, Y = self.X, self.Y

        sz = len(Y) // n_test
        x_train = np.concatenate([X[:choosen * sz, :], X[(choosen * sz + sz):, :]])
        y_train = np.concatenate([Y[:choosen * sz], Y[(choosen * sz + sz):]])
        x_val = X[choosen * sz:(choosen * sz + sz), :]
        y_val = Y[choosen * sz:(choosen * sz + sz)]

        # balance the 1 class
        x0, y0 = x_train[y_train != 1, :], y_train[y_train != 1]
        x1 = x_train[y_train == 1, :]
        x1 = np.repeat(x1, 9, axis=0)
        x_train = np.vstack([x0, x1])
        y_train = np.concatenate((y0, [1] * len(x1)))

        n, d = x_train.shape
        d = int(math.sqrt(d))
        x_train = x_train.reshape(-1, 1, d, d)
        x_val = x_val.reshape(-1, 1, d, d)

        # shuffle again
        x_train, y_train = shuffle(x_train, y_train)
        x_train = x_train.astype(np.float32)

        return x_train, y_train, x_val, y_val

    def __read_data(self, file_name):
        # images are 48x48 = 2304 size vectors
        Y = []
        X = []
        for index, line in enumerate(open(file_name)):
            if index > 0:
                row = line.split(',')
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split()])

        X, Y = np.array(X) / 255.0, np.array(Y)

        return X, Y


class HiddenLayer(object):
    def __init__(self, input_size, output_size):
        W, b = init_weight_and_bias(input_size, output_size)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ConvPoolLayer(object):
    def __init__(self, map_input, map_output, filter_width=5, filter_height=5, pool_sz=(2, 2)):
        sz = (filter_width, filter_height, map_input, map_output)
        w0 = init_filter(sz, pool_sz)
        self.W = tf.Variable(w0)
        b0 = np.zeros(map_output, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.pool_sz = pool_sz
        self.params = [self.W, self.b]

    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        p1, p2 = self.pool_sz
        pool_out = tf.nn.max_pool(
            conv_out,
            ksize=[1, p1, p2, 1],
            strides=[1, p1, p2, 1],
            padding='SAME'
        )
        return tf.nn.relu(pool_out)


class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.error_rate = None

    def fit(self, x_train, y_train, x_val, y_val, lr=1e-2, momentum=0.9, regularization=1e-3, decay=0.99999,
            batch_sz=30, epochs=5, show_fig=True):
        lr = np.float32(lr)
        momentum = np.float32(momentum)
        regularization = np.float32(regularization)
        decay = np.float32(decay)
        n_label = len(set(y_train))

        # make a validation set
        y_train = to_one_hot_encoded(y_train).astype(np.float32)
        y_val = to_one_hot_encoded(y_val).astype(np.float32)

        n_train, width, height, c = x_train.shape

        # initialize convpool layers
        self.convpool_layers, out_width, out_height = self.__create_conv_layers(width, height, c)

        # initialize hidden layers
        hidden_input_size = self.convpool_layer_sizes[-1][0] * out_width * out_height
        self.hidden_layers, hidden_size = self.__create_hidden_layers(hidden_input_size)

        # logistic regression layer
        W, b = init_weight_and_bias(hidden_size, n_label)
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)

        # collect params
        self.params = self.__get_params()

        costs = self.__train(x_train, y_train, x_val, y_val, batch_sz, lr, regularization, decay, momentum, epochs)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def __train(self, x_train, y_train, x_val, y_val, batch_sz, lr, regularization, decay, momentum, epochs):
        n_train, width, height, session_cost = x_train.shape
        y_valid_reserved = np.argmax(y_val, axis=1)
        n_label = len(set(y_valid_reserved))

        tf_x = tf.placeholder(tf.float32, shape=(None, width, height, session_cost), name='X')
        tf_y = tf.placeholder(tf.float32, shape=(None, n_label), name='Y')
        activation = self.forward(tf_x)

        regularization_cost = regularization * sum([tf.nn.l2_loss(p) for p in self.params])
        cost = regularization_cost + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=tf_y))
        prediction = self.predict(tf_x)

        train_optimizer = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=momentum).minimize(cost)

        n_batches = n_train // batch_sz
        costs = []

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            for epoch in range(epochs):
                x_train, y_train = shuffle(x_train, y_train)
                for batch in range(n_batches):
                    x_batch = x_train[batch * batch_sz:(batch * batch_sz + batch_sz)]
                    y_batch = y_train[batch * batch_sz:(batch * batch_sz + batch_sz)]

                    session.run(train_optimizer, feed_dict={tf_x: x_batch, tf_y: y_batch})

                    if batch % 20 == 0:
                        session_cost = session.run(cost, feed_dict={tf_x: x_val, tf_y: y_val})
                        costs.append(session_cost)

                        session_prediction = session.run(prediction, feed_dict={tf_x: x_val, tf_y: y_val})
                        error = error_rate(y_valid_reserved, session_prediction)
                        self.error_rate = error if self.error_rate is None else min(self.error_rate, error)
                        print("epoch:", epoch, "batch:", batch, "cost:", session_cost, "error rate:", error)

        return costs

    def __create_hidden_layers(self, hidden_size):
        hidden_layers = []
        for hidden_output in self.hidden_layer_sizes:
            h = HiddenLayer(hidden_size, hidden_output)
            hidden_layers.append(h)
            hidden_size = hidden_output

        return hidden_layers, hidden_size

    def __create_conv_layers(self, width, height, c, max_pool_sz=(2, 2)):
        map_input = c
        out_width = width
        out_height = height
        convolutional_pool_layers = []
        for map_output, filter_width, filter_height in self.convpool_layer_sizes:
            layer = ConvPoolLayer(map_input, map_output, filter_width, filter_height, max_pool_sz)
            convolutional_pool_layers.append(layer)
            out_width = out_width // max_pool_sz[0]
            out_height = out_height // max_pool_sz[1]
            map_input = map_output

        return convolutional_pool_layers, out_width, out_height

    def score(self):
        return self.error_rate

    def forward(self, x):
        next_layer = x
        for layer in self.convpool_layers:
            next_layer = layer.forward(next_layer)

        hidden_shape = next_layer.get_shape().as_list()
        next_layer = tf.reshape(next_layer, [-1, np.prod(hidden_shape[1:])])

        for layer in self.hidden_layers:
            next_layer = layer.forward(next_layer)

        return tf.matmul(next_layer, self.W) + self.b

    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, 1)

    def __get_params(self):
        params = [self.W, self.b]
        for layer in self.convpool_layers:
            params += layer.params
        for layer in self.hidden_layers:
            params += layer.params

        return params


def cross_validation():
    data_processor = DataProcessor('fer2013.csv')
    n_test = 5
    errors = []
    for k in range(n_test):
        x_train, y_train, x_val, y_val = data_processor.process_data(n_test, choosen=k)

        # reshape X for tf: N x H x W x C
        x_train = x_train.transpose((0, 2, 3, 1))
        x_val = x_val.transpose((0, 2, 3, 1))

        model = CNN(convpool_layer_sizes=[(20, 3, 3), (20, 3, 3)], hidden_layer_sizes=[256, 256], )

        model.fit(x_train, y_train, x_val, y_val)

        error = model.score()
        errors.append(error)

    return np.mean(errors)


if __name__ == '__main__':
    error_model = cross_validation()

    print('Error = ', error_model)
