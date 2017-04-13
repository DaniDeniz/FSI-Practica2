import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
data_set = cPickle.load(f)
f.close()

train_set = data_set[:int(len(data_set)*0.7)]
valid_set = data_set[int(len(data_set)*0.7):int(len(data_set)*0.85)]
test_set = data_set[int(len(data_set)*0.85):]

x_data_tr, y_data_tr = train_set
x_data_val, y_data_val = valid_set
x_data_test, y_data_test = test_set

y_data_tr = one_hot(y_data_tr,10)
y_data_val = one_hot(y_data_val,10)
y_data_test = one_hot(y_data_test,10)

# ---------------- Visualizing some element of the MNIST dataset --------------

#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!
x = tf.placeholder("float", [None, 28*28])
y_ = tf.placeholder("float", [None, 10])

W1 = tf.Variable(np.float32(np.random.rand(28*28,10*10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10*10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10*10,10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
epoch = 0
marginError = 0.0001
previousError = 10000

while 1:
    for jj in xrange(len(x_data_tr) / batch_size):
        batch_xs = x_data_tr[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_tr[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: x_data_val, y_: y_data_val})
    epoch += 1
    print "Error: ", error, " Epocas:", epoch
    if abs(error - previousError) <= marginError:
        break

    previousError = error
print "------------------------Test Set--------------------------------------------------"

result = sess.run(y, feed_dict={x: x_data_test})
coincidencias = 0.0
for b, r in zip(y_data_test, result):
    if b.argmax() == r.argmax():
        coincidencias += 1

error = sess.run(loss, feed_dict={x: x_data_test, y_: y_data_test})

print 'Error = ', error, 'Epocas = ', epoch, ' aciertos = ', coincidencias
print 'Porcentaje de acierto = ', int((coincidencias/len(y_data_test))*100), '%'

