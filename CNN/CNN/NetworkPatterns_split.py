'''
Created on 2018/03/12

@author: funabashi
'''

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
                self.x_conv, self.W_conv1, [1,2,2,1], 'VALID') + self.b_conv1)
#        Pooling Layer1
#         with tf.name_scope("pool1"):
#             self.h_pool1 = self.max_pool_2x2(self.h_conv1, kernel=[1,2,2,1], stride=[1,2,2,1], pad='SAME')
#         Convolution Layer2
        with tf.name_scope("conv2"):
            self.W_conv2 = self.weight_variable([2,2,14,28])
            self.b_conv2 = self.bias_variable([28])
            self.h_conv2 = tf.nn.relu(self.conv2d(
                self.h_conv1, self.W_conv2, [1,2,2,1], 'VALID') + self.b_conv2)
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
            for d in self.h_conv2.get_shape()[1:].as_list():
                dim *= d
            self.W_fc1 = self.weight_variable([dim+int(inputs.shape[-1]), 28])
            self.b_fc1 = self.bias_variable([28])
            self.h_pool2_flat = tf.reshape(self.h_conv2, [-1,dim]) # Make matrix into vector
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
                self.h_conv111, self.W_conv2, [1,2,2,1], 'VALID') + self.b_conv2)
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
            for d in self.h_conv2.get_shape()[1:].as_list():
                dim *= d
            self.W_fc1 = self.weight_variable([dim+int(inputs.shape[-1]), 28])
            self.b_fc1 = self.bias_variable([28])
            self.h_pool2_flat = tf.reshape(self.h_conv2, [-1,dim]) # Make matrix into vector
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
            for d in self.h_conv2.get_shape()[1:].as_list():
                dim *= d
            self.W_fc1 = self.weight_variable([dim, 28])
            self.b_fc1 = self.bias_variable([28])
            self.h_pool2_flat = tf.reshape(self.h_conv2, [-1,dim]) # Make matrix into vector
            # Concatenate featured images and other inputs
#             self.h_concatenated = tf.concat([self.h_pool2_flat,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer1
        with tf.name_scope("fc11"):
            dim = 1
            for d in self.h_conv22.get_shape()[1:].as_list():
                dim *= d
            self.W_fc1 = self.weight_variable([dim, 28])
            self.b_fc1 = self.bias_variable([28])
            self.h_pool2_flat = tf.reshape(self.h_conv22, [-1,dim]) # Make matrix into vector
            # Concatenate featured images and other inputs
#             self.h_concatenated = tf.concat([self.h_pool2_flat,inputs],1) 
            self.h_fc11 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        self.h_fc111 = tf.concat([self.h_fc1, self.h_fc11],1)
        self.h_fc111 = tf.concat([self.h_fc111, inputs],1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([72, 16])
            self.b_fc2 = self.bias_variable([16])
            self.h_fc2 = tf.matmul(self.h_fc111, self.W_fc2) + self.b_fc2
#         Fully Connected Layer3
#         with tf.name_scope("fc3"):
#             self.W_fc3 = self.weight_variable([40, 16])
#             self.b_fc3 = self.bias_variable([16])
#             self.h_fc3 = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3
        h = self.h_fc2
        return h
