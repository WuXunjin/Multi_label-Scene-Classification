import json
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime
import cv2
import math
import copy
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.chdir('F:\\Scene Classification')

## This reads the labels from the json file and stores the multi-labels as a list of lists.
f = open('labels.json').read()
l = f.split(sep ='\n')
labels = []
for x in l:
    labels.append(eval(x))


## reading the images:
path,dirs,files = os.walk("F:\\Scene Classification\\image_data").__next__()
file_count = len(files)

images = []
for i in range(file_count):
    l = cv2.imread(path+"\\"+files[i])
    gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray,(128,128))  #28,28
    images.append(img)
    
#plt.imshow(images[2])
#plt.show()

images = np.array(images)
labels = np.array(labels)

images = images.astype('float32')
labels = labels.astype('float32') #int32

## Following the Approach of using the multi-hot encoded label vectors.
## Converting the label vectors in a format that the presence of that label is denoted as 1, and the rest labels are 
## converted to 0 otherwise
labels[labels == -1] = 0

# Normalize the color values to 0-1 as the pixel values are from 0-255)
images /= 255

# Adding a colour channel dimension, which will be required later when using the CNN
# The CNN we'll use later expects a color channel dimension, (since its a greyscale image, so the colour dimension = 1,
# if it were RGB, then color dimension = 3.

images = images.reshape(images.shape[0], 128, 128, 1)

## splitting the images to train and test images
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)


if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

tf.reset_default_graph()

n_epochs = 1000
batch_size = 250
n_hidden = 300
n_outputs = 5
m = x_train.shape[0]
n_batches = int(np.ceil(m / batch_size))


X = tf.placeholder(tf.float32, shape=(None, 128,128,1), name='X')
# tf.reshape(X,[-1,128,128,1])
y = tf.placeholder(tf.float32, shape=(None,5), name='y')


## Building the CNN architecture

## tf.layers.* is the same as tf.nn.* (except for the filters and filter parameter respectively, infact tf.layers.* calls tf.nn.* at the back-end
## usually use tf.layers is used when building a model from the scratch.

with tf.name_scope('cnn'):
    conv1 = tf.layers.conv2d(inputs= tf.convert_to_tensor(X), filters=32, kernel_size=[3, 3],padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # The 'images' are now 32x32 (128 / 2 / 2), and we have 64 channels(filters) per image
    ## flattening the layer:
    pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units = n_hidden, activation=tf.nn.relu) ## units = 300 neurons
    
    ## adding dropout to the CNN:
    dropout = tf.layers.dropout(inputs=dense, rate=0.2) 
    
    #The output layer will essentially return the logits values for the 5 nuerons of the output layer.
    #output layer:
    output = tf.layers.dense(dropout, n_outputs, name='output', reuse = None)  # reuse = True

## softmax takes the logits values and squishes/normalizes the logits, such that the sum of the probabilities of the output layer is 1. 
## This is a multi-label problem, so cannot use softmax over here.
## So, we use a sigmoidal cross entropy losss function, which will give independent probabilities for each class,
## so that sum of the probabilites of all the labels/classes are not necessarily constrained to summing to 1.


## So effectively, we have one sigmoid output for each label and we minimize the (binary) cross-entropy for each label
with tf.name_scope('loss'):
    #xentropy = tf.nn.softmax_cross_entropy_with_logits(labels =y, logits = output)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = output)
    loss = tf.reduce_mean(xentropy, name='loss')
    loss_summary = tf.summary.scalar('log_loss', loss)
    
learningrate = 0.001

## backpropogation using the Grad Descent Optimizer
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learningrate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope('eval'):
    # This gives % of images where ALL labels are predicted correctly.
    # As output is in form of logits, so need to convert it through the sigmoid function.
    predictions = tf.round(tf.nn.sigmoid(output))
    correct = tf.equal(predictions,tf.round(y))
    #correct = tf.equal(tf.argmax(output,1), tf.argmax(y, 1))    - this would work if it were 1 label per image

    ## Mean accuracy over all the labels:
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    
    ## Accuracy, where all labels need to be correct:
    all_labels_true = tf.reduce_min(tf.cast(correct, tf.float32), 1)
    accuracy2 = tf.reduce_mean(all_labels_true)
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def log_dir(prefix=''):
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = './tf_logs'
    if prefix:
        prefix += '-'
    name = prefix + 'run-' + now
    return '{}/{}/'.format(root_logdir, name)


logdir = log_dir('Scene Classification')

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

## This function is used to get the next batch
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    indices = list(indices)
    # print(indices)
    X_batch = x_train[indices]
    y_batch = y_train[indices]
    return X_batch, y_batch

checkpoint_path = 'F:\\Scene Classification\\multi_label_assignment.ckpt'
checkpoint_epoch_path = checkpoint_path + '.epoch'
final_model_path = '.\\multi_label_assignemnt'

best_loss = np.infty
epochs_without_progress = 0
max_epoch_without_progress = 50

## Optimizing Tensorflow for CPU:
config = tf.ConfigProto()
## Utilizing the inter and intra threads available
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44

##Utilizing the XLA(Accelerated Linear Algebra) for building the tensorflow graphs
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

## running the tensorflow graph by creating a tensorflow session and running the operations.
## the main operation to call is the 'training_op', which will initiate the training, 
with tf.Session() as sess:
    sess.config = config
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, 'rb') as f:
            start_epoch = int(f.read())
        print('Training was interrupted start epoch at:', start_epoch)
        saver.restore(sess, checkpoint_path)

    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for iteration in range(x_train.shape[0] // batch_size):
            X_batch, y_batch = fetch_batch(epoch, iteration, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str, correct_val = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary, correct], feed_dict={X: x_test, y: y_test})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        acc_train = accuracy.eval(feed_dict={X: x_test,y: y_test})
        
        
        if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation Accuracy : {:.3f}".format(accuracy_val * 100),
                  "\t Train accuracy :{}".format(acc_train * 100),
                  "\tLoss :{:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, 'wb') as f:
                f.write(b'%d' % (epoch + 1))
            # if loss_val < best_loss:
            #     saver.save(sess, final_model_path)
            #     best_loss = loss_val
            # else:
            #     epochs_without_progress += 5
            #     if epochs_without_progress > max_epoch_without_progress:
            #         print('early stopping at epoch', epoch)
            #         break
    
    ## This gives the predicted values/labels for the test data
    predictions = sess.run(tf.round(tf.nn.sigmoid(output)), feed_dict={X:x_test})

os.remove(checkpoint_epoch_path)