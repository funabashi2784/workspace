'''
Created on 2017/12/29

@author: funabashi
'''
#concatenate at fc2
#Thumb
        self.x_conv_t = tf.reshape(inputs_t, [-1, 4, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1_t"):
            self.W_conv1_t = self.weight_variable([2,2,3,7])
            self.b_conv1_t = self.bias_variable([7])
            self.h_conv1_t = tf.nn.relu(self.conv2d(self.x_conv_t, self.W_conv1_t) + self.b_conv1_t)
#        Pooling Layer1
        with tf.name_scope("pool1_t"):
            self.h_pool1_t = self.max_pool_2x2(self.h_conv1_t)
#         Convolution Layer2
        with tf.name_scope("conv2_t"):
            self.W_conv2_t = self.weight_variable([2,2,7,14])
            self.b_conv2_t = self.bias_variable([14])
            self.h_conv2_t = tf.nn.relu(self.conv2d(self.h_pool1_t, self.W_conv2_t) + self.b_conv2_t)
#        Pooling Layer2
        with tf.name_scope("pool2_t"):
            self.h_pool2_t = self.max_pool_2x2(self.h_conv2_t)
#         Fully Connected Layer1
        with tf.name_scope("fc1_t"):
            dim1 = 1
            for d in self.h_pool2_t.get_shape()[1:].as_list():
                dim1 *= d
            self.h_pool2_flat_t = tf.reshape(self.h_pool2_t, [-1,dim1]) # Make matrix into vector
            self.W_fc1_i = self.weight_variable([dim1, 50])
            self.b_fc1_i = self.bias_variable([50])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat_t, self.W_fc1) + self.b_fc1)
#Index
        self.x_conv_i = tf.reshape(inputs_i, [-1, 4, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1_i"):
            self.W_conv1_i = self.weight_variable([2,2,3,7])
            self.b_conv1_i = self.bias_variable([7])
            self.h_conv1_i = tf.nn.relu(self.conv2d(self.x_conv_i, self.W_conv1_i) + self.b_conv1_i)
#        Pooling Layer1
        with tf.name_scope("pool1_i"):
            self.h_pool1_i = self.max_pool_2x2(self.h_conv1_i)
#         Convolution Layer2
        with tf.name_scope("conv2_i"):
            self.W_conv2_i = self.weight_variable([2,2,7,14])
            self.b_conv2_i = self.bias_variable([14])
            self.h_conv2_i = tf.nn.relu(self.conv2d(self.h_pool1_i, self.W_conv2_i) + self.b_conv2_i)
#        Pooling Layer2
        with tf.name_scope("pool2_i"):
            self.h_pool2_i = self.max_pool_2x2(self.h_conv2_i)
#         Fully Connected Layer1
        with tf.name_scope("fc1_i"):
            dim1 = 1
            for d in self.h_pool2_t.get_shape()[1:].as_list():
                dim1 *= d
            self.h_pool2_flat_t = tf.reshape(self.h_pool2_t, [-1,dim1]) # Make matrix into vector
            dim2 = 1
            for d in self.h_pool2_i.get_shape()[1:].as_list():
                dim2 *= d
            self.W_fc1_i = self.weight_variable([dim1+dim2, 50])
            self.b_fc1_i = self.bias_variable([50])
            self.h_pool2_flat_i = tf.reshape(self.h_pool2_i, [-1,dim2]) # Make matrix into vector
            self.h_fc1_i = tf.nn.relu(tf.matmul(self.h_pool2_flat_t, self.W_fc1_i) + self.b_fc1_i)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([50, 16])
            self.b_fc2 = self.bias_variable([16])
            self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
            # Concatenate featured images and other inputs
            self.h_concatenated_ti = tf.concat([self.h_pool2_flat_t,self.h_pool2_flat_i],1)
            self.h_concatenated = tf.concat([self.h_concatenated_ti,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
        h = self.h_fc2
        
#concatenate at fc1
#Thumb
        self.x_conv_t = tf.reshape(inputs_t, [-1, 4, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1_t"):
            self.W_conv1_t = self.weight_variable([2,2,3,7])
            self.b_conv1_t = self.bias_variable([7])
            self.h_conv1_t = tf.nn.relu(self.conv2d(self.x_conv_t, self.W_conv1_t) + self.b_conv1_t)
#        Pooling Layer1
        with tf.name_scope("pool1_t"):
            self.h_pool1_t = self.max_pool_2x2(self.h_conv1_t)
#         Convolution Layer2
        with tf.name_scope("conv2_t"):
            self.W_conv2_t = self.weight_variable([2,2,7,14])
            self.b_conv2_t = self.bias_variable([14])
            self.h_conv2_t = tf.nn.relu(self.conv2d(self.h_pool1_t, self.W_conv2_t) + self.b_conv2_t)
#        Pooling Layer2
        with tf.name_scope("pool2_t"):
            self.h_pool2_t = self.max_pool_2x2(self.h_conv2_t)
#Index
        self.x_conv_i = tf.reshape(inputs_i, [-1, 4, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1_i"):
            self.W_conv1_i = self.weight_variable([2,2,3,7])
            self.b_conv1_i = self.bias_variable([7])
            self.h_conv1_i = tf.nn.relu(self.conv2d(self.x_conv_i, self.W_conv1_i) + self.b_conv1_i)
#        Pooling Layer1
        with tf.name_scope("pool1_i"):
            self.h_pool1_i = self.max_pool_2x2(self.h_conv1_i)
#         Convolution Layer2
        with tf.name_scope("conv2_i"):
            self.W_conv2_i = self.weight_variable([2,2,7,14])
            self.b_conv2_i = self.bias_variable([14])
            self.h_conv2_i = tf.nn.relu(self.conv2d(self.h_pool1_i, self.W_conv2_i) + self.b_conv2_i)
#        Pooling Layer2
        with tf.name_scope("pool2_i"):
            self.h_pool2_i = self.max_pool_2x2(self.h_conv2_i)
#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            dim1 = 1
            for d in self.h_pool2_t.get_shape()[1:].as_list():
                dim1 *= d
            self.h_pool2_flat_t = tf.reshape(self.h_pool2_t, [-1,dim1]) # Make matrix into vector
            dim2 = 1
            for d in self.h_pool2_i.get_shape()[1:].as_list():
                dim2 *= d
            self.W_fc1_i = self.weight_variable([dim1+dim2, 50])
            self.b_fc1_i = self.bias_variable([50])
            self.h_pool2_flat_i = tf.reshape(self.h_pool2_i, [-1,dim2]) # Make matrix into vector
            # Concatenate featured images and other inputs
            self.h_concatenated_ti = tf.concat([self.h_pool2_flat_t,self.h_pool2_flat_i],1)
            self.h_concatenated = tf.concat([self.h_concatenated_ti,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([50, 16])
            self.b_fc2 = self.bias_variable([16])
            self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        h = self.h_fc2

#concatenate at conv2
#Thumb
        self.x_conv_t = tf.reshape(inputs_t, [-1, 4, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1_t"):
            self.W_conv1_t = self.weight_variable([2,2,3,7])
            self.b_conv1_t = self.bias_variable([7])
            self.h_conv1_t = tf.nn.relu(self.conv2d(self.x_conv_t, self.W_conv1_t) + self.b_conv1_t)
#        Pooling Layer1
        with tf.name_scope("pool1_t"):
            self.h_pool1_t = self.max_pool_2x2(self.h_conv1_t)
#Index
        self.x_conv_i = tf.reshape(inputs_i, [-1, 4, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1_i"):
            self.W_conv1_i = self.weight_variable([2,2,3,7])
            self.b_conv1_i = self.bias_variable([7])
            self.h_conv1_i = tf.nn.relu(self.conv2d(self.x_conv_i, self.W_conv1_i) + self.b_conv1_i)
#        Pooling Layer1
        with tf.name_scope("pool1_i"):
            self.h_pool1_i = self.max_pool_2x2(self.h_conv1_i)
        # Concatenate featured images and other inputs
        self.h_concatenated_ti = tf.concat([self.h_pool1_t,self.h_pool1_i],1)
#         Convolution Layer2
        with tf.name_scope("conv2_i"):
            self.W_conv2_i = self.weight_variable([2,2,7,14])
            self.b_conv2_i = self.bias_variable([14])
            self.h_conv2_i = tf.nn.relu(self.conv2d(self.h_pool1_i, self.W_conv2_i) + self.b_conv2_i)
#        Pooling Layer2
        with tf.name_scope("pool2_i"):
            self.h_pool2_i = self.max_pool_2x2(self.h_conv2_i)
#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            dim1 = 1
            for d in self.h_pool2_t.get_shape()[1:].as_list():
                dim1 *= d
            self.h_pool2_flat_t = tf.reshape(self.h_pool2_t, [-1,dim1]) # Make matrix into vector
            dim2 = 1
            for d in self.h_pool2_i.get_shape()[1:].as_list():
                dim2 *= d
            self.W_fc1_i = self.weight_variable([dim1+dim2, 50])
            self.b_fc1_i = self.bias_variable([50])
            self.h_pool2_flat_i = tf.reshape(self.h_pool2_i, [-1,dim2]) # Make matrix into vector
            # Concatenate featured images and other inputs
 
            
            
            self.h_concatenated_ti = tf.concat([self.h_pool2_flat_t,self.h_pool2_flat_i],1)
            self.h_concatenated = tf.concat([self.h_concatenated_ti,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([50, 16])
            self.b_fc2 = self.bias_variable([16])
            self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        h = self.h_fc2
         
#concatenate at conv1
#Thumb
        self.x_conv_t = tf.reshape(inputs_t, [-1, 4, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1_t"):
            self.W_conv1_t = self.weight_variable([2,2,3,7])
            self.b_conv1_t = self.bias_variable([7])
            self.h_conv1_t = tf.nn.relu(self.conv2d(self.x_conv_t, self.W_conv1_t) + self.b_conv1_t)
#        Pooling Layer1
        with tf.name_scope("pool1_t"):
            self.h_pool1_t = self.max_pool_2x2(self.h_conv1_t)
#Index
        self.x_conv_i = tf.reshape(inputs_i, [-1, 4, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1_i"):
            self.W_conv1_i = self.weight_variable([2,2,3,7])
            self.b_conv1_i = self.bias_variable([7])
            self.h_conv1_i = tf.nn.relu(self.conv2d(self.x_conv_i, self.W_conv1_i) + self.b_conv1_i)
#        Pooling Layer1
        with tf.name_scope("pool1_i"):
            self.h_pool1_i = self.max_pool_2x2(self.h_conv1_i)
        # Concatenate featured images and other inputs
        self.h_concatenated_ti = tf.concat([self.h_pool1_t,self.h_pool1_i],1)
#         Convolution Layer2
        with tf.name_scope("conv2_i"):
            self.W_conv2_i = self.weight_variable([2,2,14,28])
            self.b_conv2_i = self.bias_variable([28])
            self.h_conv2_i = tf.nn.relu(self.conv2d(self.h_concatenated_ti, self.W_conv2_i) + self.b_conv2_i)
#        Pooling Layer2
        with tf.name_scope("pool2_i"):
            self.h_pool2_i = self.max_pool_2x2(self.h_conv2_i)
#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            dim1 = 1
            for d in self.h_pool2_t.get_shape()[1:].as_list():
                dim1 *= d
            self.h_pool2_flat_t = tf.reshape(self.h_pool2_t, [-1,dim1]) # Make matrix into vector
            dim2 = 1
            for d in self.h_pool2_i.get_shape()[1:].as_list():
                dim2 *= d
            self.W_fc1_i = self.weight_variable([dim1+dim2, 50])
            self.b_fc1_i = self.bias_variable([50])
            self.h_pool2_flat_i = tf.reshape(self.h_pool2_i, [-1,dim2]) # Make matrix into vector
            self.h_concatenated = tf.concat([self.h_concatenated_ti,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([50, 16])
            self.b_fc2 = self.bias_variable([16])
            self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        h = self.h_fc2

#concatenate at input
        self.x_conv = tf.reshape(input_vis, [-1, 8, 4, 3])
#         Convolution Layer1
        with tf.name_scope("conv1"):
            self.W_conv1 = self.weight_variable([2,2,3,7])
            self.b_conv1 = self.bias_variable([7])
            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_conv, self.W_conv1) + self.b_conv1)
#        Pooling Layer1
        with tf.name_scope("pool1"):
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)
#         Convolution Layer2
        with tf.name_scope("conv2"):
            self.W_conv2 = self.weight_variable([2,2,7,14])
            self.b_conv2 = self.bias_variable([14])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
#        Pooling Layer2
        with tf.name_scope("pool2"):
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)#         Fully Connected Layer1
        with tf.name_scope("fc1"):
            dim1 = 1
            for d in self.h_pool2_t.get_shape()[1:].as_list():
                dim1 *= d
            self.h_pool2_flat_t = tf.reshape(self.h_pool2_t, [-1,dim1]) # Make matrix into vector
            dim2 = 1
            for d in self.h_pool2_i.get_shape()[1:].as_list():
                dim2 *= d
            self.W_fc1_i = self.weight_variable([dim1+dim2, 50])
            self.b_fc1_i = self.bias_variable([50])
            self.h_pool2_flat_i = tf.reshape(self.h_pool2_i, [-1,dim2]) # Make matrix into vector
            self.h_concatenated = tf.concat([self.h_concatenated_ti,inputs],1) 
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_concatenated, self.W_fc1) + self.b_fc1)
#         Fully Connected Layer2
        with tf.name_scope("fc2"):
            self.W_fc2 = self.weight_variable([50, 16])
            self.b_fc2 = self.bias_variable([16])
            self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        h = self.h_fc2