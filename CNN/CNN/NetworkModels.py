###--- Tutorial-like ---###
#        Reshape input into [28,28,1]
        self.x_conv = tf.reshape(input_vis, [-1, 28, 28, 1])
#         Convolution Layer1
        with tf.name_scope("conv1"):
            self.W_conv1 = self.weight_variable([5,5,1,32])
            self.b_conv1 = self.bias_variable([32])
            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_conv, self.W_conv1) + self.b_conv1)
#        Pooling Layer1
        with tf.name_scope("pool1"):
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)
#         Convolution Layer2
        with tf.name_scope("conv2"):
            self.W_conv2 = self.weight_variable([5,5,32,64])
            self.b_conv2 = self.bias_variable([64])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
#        Pooling Layer2
        with tf.name_scope("pool2"):
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)
#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            self.W_fc1 = self.weight_variable([7*7*64+40, 1024])
            self.b_fc1 = self.bias_variable([1024])
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1,7*7*64]) # Make matrix into vector
            # Concatenate featured images and other inputs
            self.h_concatenated = tf.concat([self.h_pool2_flat,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([1024, 16])
            self.b_fc2 = self.bias_variable([16])
            self.h_fc2 = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
        h = self.h_fc2
        
###--- VGGNet-like ---###
        self.x_conv = tf.reshape(input_vis, [-1, 96, 96, 1])
#         Convolution Layer1
        with tf.name_scope("conv1"):
            self.W_conv1 = self.weight_variable([5,5,1,32])
            self.b_conv1 = self.bias_variable([32])
            self.h_conv1 = tf.nn.relu(self.conv2d(
                self.x_conv, self.W_conv1, [1,1,1,1], 'SAME') + self.b_conv1)
#        Pooling Layer1
        with tf.name_scope("pool1"):
            self.h_pool1 = self.max_pool_2x2(self.h_conv1, [1,2,2,1], [1,2,2,1], 'SAME')
#         Convolution Layer2
        with tf.name_scope("conv2"):
            self.W_conv2 = self.weight_variable([5,5,32,64])
            self.b_conv2 = self.bias_variable([64])
            self.h_conv2 = tf.nn.relu(self.conv2d(
                self.h_pool1, self.W_conv2, [1,1,1,1], 'SAME') + self.b_conv2)
#        Pooling Layer2
        with tf.name_scope("pool2"):
            self.h_pool2 = self.max_pool_2x2(self.h_conv2, [1,2,2,1], [1,2,2,1], 'SAME')
#         Convolution Layer3
        with tf.name_scope("conv3"):
            self.W_conv3 = self.weight_variable([5,5,64,128])
            self.b_conv3 = self.bias_variable([128])
            self.h_conv3 = tf.nn.relu(self.conv2d(
                self.h_pool2, self.W_conv3, [1,1,1,1], 'SAME') + self.b_conv3)
#        Pooling Layer2
        with tf.name_scope("pool2"):
            self.h_pool3 = self.max_pool_2x2(self.h_conv3, [1,2,2,1], [1,2,2,1], 'SAME')
#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            dim = 1
            for d in self.h_pool3.get_shape()[1:].as_list():
                dim *= d
            self.W_fc1 = self.weight_variable([dim+int(inputs.shape[-1]), 1024])
            self.b_fc1 = self.bias_variable([1024])
            self.h_pool2_flat = tf.reshape(self.h_pool3, [-1,dim]) # Make matrix into vector
            # Concatenate featured images and other inputs
            self.h_concatenated = tf.concat([self.h_pool2_flat,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([1024, 256])
            self.b_fc2 = self.bias_variable([256])
            self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
            self.h_fc2_drop = tf.nn.dropout(self.h_fc2, keep_prob)
#         Fully Connected Layer3
        with tf.name_scope("fc3"):
            self.W_fc3 = self.weight_variable([256, 16])
            self.b_fc3 = self.bias_variable([16])
            self.h_fc3 = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3
        h = self.h_fc3


###--- Deep Visio-Motor like ---###
        self.x_conv = tf.reshape(input_vis, [-1, 96, 96, 1])
#         Convolution Layer1
        with tf.name_scope("conv1"):
            self.W_conv1 = self.weight_variable([7,7,1,64])
            self.b_conv1 = self.bias_variable([64])
            self.h_conv1 = tf.nn.relu(self.conv2d(
                self.x_conv, self.W_conv1, [1,1,1,1], 'SAME') + self.b_conv1)
#         Convolution Layer2
        with tf.name_scope("conv2"):
            self.W_conv2 = self.weight_variable([5,5,64,32])
            self.b_conv2 = self.bias_variable([32])
            self.h_conv2 = tf.nn.relu(self.conv2d(
                self.h_conv1, self.W_conv2, [1,1,1,1], 'SAME') + self.b_conv2)
#         Convolution Layer3
        with tf.name_scope("conv3"):
            self.W_conv3 = self.weight_variable([5,5,32,32])
            self.b_conv3 = self.bias_variable([32])
            self.h_conv3 = tf.nn.relu(self.conv2d(
                self.h_conv2, self.W_conv3, [1,1,1,1], 'SAME') + self.b_conv3)
#         Convolution Layer4
        with tf.name_scope("conv4"):
            self.W_conv4 = self.weight_variable([3,3,128,256])
            self.b_conv4 = self.bias_variable([256])
            self.h_conv4 = tf.nn.softmax(self.conv2d(
                self.h_conv3, self.W_conv4, [1,1,1,1], 'SAME') + self.b_conv4)
#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            dim = 1
            for d in self.h_conv4.get_shape()[1:].as_list():
                dim *= d
            self.W_fc1 = self.weight_variable([dim+int(inputs.shape[-1]), 64])
            self.b_fc1 = self.bias_variable([64])
            self.h_pool2_flat = tf.reshape(self.h_conv4, [-1,dim]) # Make matrix into vector
            # Concatenate featured images and other inputs
            self.h_concatenated = tf.concat([self.h_pool2_flat,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([64, 40])
            self.b_fc2 = self.bias_variable([40])
            self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
#         Fully Connected Layer3
        with tf.name_scope("fc3"):
            self.W_fc3 = self.weight_variable([40, 16])
            self.b_fc3 = self.bias_variable([16])
            self.h_fc3 = tf.matmul(self.h_fc2, self.W_fc2) + self.b_fc2
        h = self.h_fc3


###--- Deep Visio-Motor like ---###
        self.x_conv = tf.reshape(input_vis, [-1, 240, 240, 1])
#         Convolution Layer1
        with tf.name_scope("conv1"):
            self.W_conv1 = self.weight_variable([7,7,1,64])
            self.b_conv1 = self.bias_variable([64])
            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_conv, self.W_conv1) + self.b_conv1)
#         Convolution Layer2
        with tf.name_scope("conv2"):
            self.W_conv2 = self.weight_variable([5,5,64,32])
            self.b_conv2 = self.bias_variable([32])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)
#         Convolution Layer3
        with tf.name_scope("conv3"):
            self.W_conv3 = self.weight_variable([5,5,32,32])
            self.b_conv3 = self.bias_variable([32])
            self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3) + self.b_conv3)
#         Convolution Layer4
        with tf.name_scope("conv4"):
            self.W_conv4 = self.weight_variable([3,3,128,256])
            self.b_conv4 = self.bias_variable([256])
            self.h_conv4 = tf.nn.softmax(self.conv2d(self.h_conv3, self.W_conv4) + self.b_conv4)
#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            dim = 1
            for d in self.h_pool4.get_shape()[1:].as_list():
                dim *= d
            self.W_fc1 = self.weight_variable([dim+int(inputs.shape[-1]), 64])
            self.b_fc1 = self.bias_variable([64])
            self.h_pool2_flat = tf.reshape(self.h_conv4, [-1,dim]) # Make matrix into vector
            # Concatenate featured images and other inputs
            self.h_concatenated = tf.concat([self.h_pool2_flat,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([64, 40])
            self.b_fc2 = self.bias_variable([40])
            self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
#         Fully Connected Layer3
        with tf.name_scope("fc3"):
            self.W_fc3 = self.weight_variable([40, 16])
            self.b_fc3 = self.bias_variable([16])
            self.h_fc3 = tf.matmul(self.h_fc2, self.W_fc2) + self.b_fc3
        h = self.h_fc3