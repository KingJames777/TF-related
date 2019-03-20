from second import *

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(49000, 1000, 0)

learning_rate = 1e-3
decay_every = 300
decay = 0.95
regularization = 1e-3

X = tf.placeholder('float32', [None, 32, 32, 3])
y = tf.placeholder('int64', [None])
is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0)

W_conv1 = tf.get_variable('W_conv1', [5, 5, 3, 64]) 
b_conv1 = tf.get_variable('b_conv1', [64])

W_conv2 = tf.get_variable('W_conv2', [5, 5, 64, 64]) 
b_conv2 = tf.get_variable('b_conv2', [64])

W_conv3 = tf.get_variable('W_conv3', [5, 5, 64, 64])
b_conv3 = tf.get_variable('b_conv3', [64])

W1 = tf.get_variable('W1', [4096, 1024])
b1 = tf.get_variable('b1', [1024])

W2 = tf.get_variable('W2', [1024, 256])
b2 = tf.get_variable('b2', [256])

W3 = tf.get_variable('W3', [256, 10])
b3 = tf.get_variable('b3', [10])


'''  [ conv - bn - relu - pool ] * 2 - conv - bn - relu - affine - relu - affine - relu - affine - softmax  '''

conv1 = tf.nn.conv2d(X, W_conv1, [1,1,1,1], 'SAME') + b_conv1  
bn1 = tf.layers.batch_normalization(conv1,training = is_training)
relu1 = tf.nn.relu(bn1)  ## pooling 2*2, stride 2
pool1 = tf.nn.max_pool(relu1, [1,2,2,1], [1,2,2,1], 'VALID')  

conv2 = tf.nn.conv2d(pool1, W_conv2, [1,1,1,1], 'SAME') + b_conv2  
bn2 = tf.layers.batch_normalization(conv2,training = is_training)
relu2 = tf.nn.relu(bn2)
pool2 = tf.nn.max_pool(relu2, [1,2,2,1], [1,2,2,1], 'VALID')

conv3 = tf.nn.conv2d(pool2, W_conv3, [1,1,1,1], 'SAME') + b_conv3  
bn3 = tf.layers.batch_normalization(conv3,training = is_training)
relu3 = tf.nn.relu(bn3)

fc1 = tf.nn.relu(tf.matmul(tf.reshape(relu3, [-1, 4096]), W1) + b1)
fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
y_out = tf.matmul(fc2, W3) + b3

## loss and optimizer
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(tf.one_hot(y,10), y_out))
mean_loss += regularization * (tf.nn.l2_loss(W2) + tf.nn.l2_loss(W1) + tf.nn.l2_loss(W3) +
                               tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv1))

lr = tf.train.exponential_decay(learning_rate, global_step, decay_every, decay, True)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  ## Keep in mind!!!
with tf.control_dependencies(extra_update_ops):
    optimizer = tf.train.AdamOptimizer(lr).minimize(mean_loss, global_step = global_step)

def run_model(session, predict, loss_val, X_, y_, epochs=1, batch_size=64,
              print_every=100, training=None, plot_losses=False):
    
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_train = X_.shape[0]
    num_batch_per_epoch = num_train//batch_size
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, correct_prediction, accuracy]
    training_now = training is not None
    if training_now:
        variables[-1] = training  ## optimizer
        
        losses = []
        iter_cnt = 0
        for e in range(epochs):
            for i in range(num_batch_per_epoch):
                
                # Make a minibatch of training data
                batch_mask = np.random.choice(num_train, batch_size)
                X_batch = X_[batch_mask]
                y_batch = y_[batch_mask]
                
                feed_dict = {X: X_batch, y: y_batch, is_training: training_now }
                loss, corr, _ = session.run(variables, feed_dict)
                    
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt, loss, np.sum(corr)/batch_size))
                    
                iter_cnt += 1
                losses.append(loss)
    else:  ## valiadation/test
        feed_dict = {X: X_, y: y_, is_training: training_now }
        loss, corr, acc = session.run(variables, feed_dict)
        print('loss = {0:.2g}, accuracy = {1:.2f}'.format(loss, acc))
        
    if plot_losses:
        plt.plot(losses)
        plt.show()
    return loss, corr

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run_model(sess, y_out, mean_loss, X_train, y_train, 3, 64, 200, optimizer, True)
    run_model(sess, y_out, mean_loss, X_val, y_val)
##    run_model(sess,y_out,mean_loss,X_test,y_test)
