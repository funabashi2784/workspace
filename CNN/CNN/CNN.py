import numpy as np
import tensorflow as tf
from IPython.core.debugger import Tracer; keyboard = Tracer()
#===============================================================================
# Network Making
#===============================================================================
class CNN(object):
    def __init__(self, sess,indata, targetdata,indata_tactile, 
                 learning_rate=0.0001,batch_norm_flag=False, summary=None, log_dir=None, seed=1):
        self.sess = sess
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.learning_rate = learning_rate
        self.batch_norm_flag = batch_norm_flag
        self.summary = summary
        # Make network parameters for building network in advance
        self.input_placeholder = self.placeholder(indata)
        self.teacher_placeholder = self.placeholder(targetdata)
        self.input_placeholder_tac = self.placeholder(indata_tactile)
        self.keep_prob_placeholder = tf.placeholder("float")     # dropout率を入れる仮のTensor
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.output = self.forward(inputs=self.input_placeholder, 
            input_tac=self.input_placeholder_tac, keep_prob=self.keep_prob_placeholder)
        self.loss = self.calcLoss(target=self.teacher_placeholder, output=self.output)
        self.update_params = self.optimize(self.loss, learning_rate)
        self.accuracy = self.get_accuracy(self.output,self.teacher_placeholder)
        # should be defined after all variables are defined
        self.saver = tf.train.Saver() 
        self.summary_writer = self.summaryFW(log_dir, self.sess.graph)

### Definition of placeholder & variables ###
    def placeholder(self, data):
        return tf.placeholder(tf.float32, shape=(None, data.shape[1]))
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def conv2d(self, x, W, stride, pad):
        return tf.nn.conv2d(x, W, strides=stride, padding=pad)
    def max_pool_2x2(self, x, kernel, stride, pad):
        return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding=pad)
    def batch_norm(self,x, is_training, decay=0.9, eps=1e-5):
        shape = x.get_shape().as_list()
        assert len(shape) in [2, 4]
        n_out = shape[-1]
        beta = tf.Variable(tf.zeros([n_out]))
        gamma = tf.Variable(tf.ones([n_out]))
        if len(shape) == 2:
            batch_mean, batch_var = tf.nn.moments(x, [0])
        else:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(is_training, mean_var_with_update,lambda : (ema.average(batch_mean), ema.average(batch_var)))
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

### Build Network ###
    def forward(self, inputs, input_tac, keep_prob):
        input_tac1 = input_tac[:,:90]
        input_tac2 = input_tac[:,90:]
        input_tac1 = tf.reshape(input_tac1, [-1, 6, 5, 3])
        input_tac2 = tf.reshape(input_tac2, [-1, 6, 5, 3])
        self.x_conv = tf.concat([input_tac1, input_tac2],2)
#         Convolution Layer1
        with tf.name_scope("conv1"):
            self.W_conv1 = self.weight_variable([2,2,3,14])
            self.b_conv1 = self.bias_variable([14])
            self.h_conv1 = tf.nn.relu(self.conv2d(
                input_tac1, self.W_conv1, [1,2,2,1], 'VALID') + self.b_conv1)
#         Convolution Layer1
        with tf.name_scope("conv11"):
            self.W_conv1 = self.weight_variable([2,2,3,14])
            self.b_conv1 = self.bias_variable([14])
            self.h_conv11 = tf.nn.relu(self.conv2d(
                input_tac2, self.W_conv1, [1,2,2,1], 'VALID') + self.b_conv1)
        self.h_conv111 = tf.concat([self.h_conv1, self.h_conv11],2)
#        Pooling Layer1
#         with tf.name_scope("pool1"):
#             self.h_pool1 = self.max_pool_2x2(self.h_conv1, kernel=[1,2,2,1], stride=[1,2,2,1], pad='SAME')
#         Convolution Layer2
        with tf.name_scope("conv2"):
            self.W_conv2 = self.weight_variable([2,2,14,28])
            self.b_conv2 = self.bias_variable([28])
            self.h_conv2 = tf.nn.relu(self.conv2d(
                self.h_conv1, self.W_conv2, [1,2,2,1], 'VALID') + self.b_conv2)
#         Convolution Layer2
        with tf.name_scope("conv22"):
            self.W_conv2 = self.weight_variable([2,2,14,28])
            self.b_conv2 = self.bias_variable([28])
            self.h_conv22 = tf.nn.relu(self.conv2d(
                self.h_conv11, self.W_conv2, [1,2,2,1], 'VALID') + self.b_conv2)
        self.h_conv222 = tf.concat([self.h_conv2, self.h_conv22],2)
#        Pooling Layer2
#         with tf.name_scope("pool2"):
#             self.h_pool2 = self.max_pool_2x2(self.h_conv2, kernel=[1,2,2,1], stride=[1,2,2,1], pad='SAME')
#         Convolution Layer3
#         with tf.name_scope("conv3"):
#             self.W_conv3 = self.weight_variable([2,2,14,28])
#             self.b_conv3 = self.bias_variable([28])
#             self.h_conv3 = tf.nn.relu(self.conv2d(
#                 self.h_conv2, self.W_conv3, [1,1,1,1], 'SAME') + self.b_conv3)
#         Convolution Layer4
#         with tf.name_scope("conv4"):
#             self.W_conv4 = self.weight_variable([3,3,32,32])
#             self.b_conv4 = self.bias_variable([32])
#             self.h_conv4 = tf.nn.softmax(self.conv2d(
#                 self.h_conv3, self.W_conv4, [1,1,1,1], 'SAME') + self.b_conv4)
#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            dim = 1
            for d in self.h_conv222.get_shape()[1:].as_list():
                dim *= d
            self.W_fc1 = self.weight_variable([dim+int(inputs.shape[-1]), 28])
            self.b_fc1 = self.bias_variable([28])
            self.h_pool2_flat = tf.reshape(self.h_conv222, [-1,dim]) # Make matrix into vector
            # Concatenate featured images and other inputs
            self.h_concatenated = tf.concat([self.h_pool2_flat,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([28, 16])
            self.b_fc2 = self.bias_variable([16])
            self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
#         Fully Connected Layer3
#         with tf.name_scope("fc3"):
#             self.W_fc3 = self.weight_variable([40, 16])
#             self.b_fc3 = self.bias_variable([16])
#             self.h_fc3 = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3
        h = self.h_fc2
        return h

#### Definition of Optimization ###
    def calcLoss(self,output,target):
        with tf.name_scope('mean_squared_error'):
            beta = 0.0001
#             mean_squared_error = tf.reduce_mean(tf.square(output - target)+ beta * tf.nn.l2_loss(self.W["W" + str(1)])) 
            mean_squared_error = tf.reduce_mean(tf.square(output - target))
            self.summary.scalar("mean_squared_error", mean_squared_error)
        return mean_squared_error
    def optimize(self, loss, learning_rate):
        with tf.name_scope('optimizer'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step
    def get_accuracy(self, output, label):
        with tf.name_scope('accuracy'):
            correctness = tf.reduce_mean(tf.cast(tf.equal(
                    tf.argmax(output,1), tf.argmax(label, 1)), tf.float32))
            self.summary.scalar('accuracy', correctness)
        return correctness

#### Manager of network models & parameters ###
    def save(self, filepath_name, step):
        self.saver.save(self.sess, filepath_name, global_step = step)
    def restore(self, step):
        self.saver.restore(self.sess, step)
    def summaryFW(self,dirpath, graph):
        return self.summary.FileWriter(dirpath, graph)

#### Run Operation ###
    def predict(self, feed_dict):
        return self.sess.run([self.output], feed_dict = feed_dict)[0]
    def losspre(self,feed_dict):
        return self.sess.run([self.loss], feed_dict = feed_dict)[0]
    def train(self, feed_dict):
        return self.sess.run([self.loss, self.update_params], feed_dict = feed_dict)
    def acc(self, feed_dict):
        return self.sess.run(self.accuracy, feed_dict = feed_dict)
    def sumrun(self, feed_dict, Summ):
        return self.sess.run([Summ], feed_dict = feed_dict)[0]