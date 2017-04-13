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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
#x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
x_data_tr = data[:int(len(data)*0.70), 0:4].astype('f4')  # the samples are the four first rows of data
x_data_val = data[int(len(data)*0.70):int(len(data)*0.85), 0:4].astype('f4')
x_data_test = data[int(len(data)*0.85):int(len(data)), 0:4].astype('f4')
y_data_tr = one_hot(data[:int(len(data)*0.70), 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
y_data_val = one_hot(data[int(len(data)*0.70):int(len(data)*0.85), 4].astype(int), 3)
y_data_test = one_hot(data[int(len(data)*0.85):int(len(data)), 4].astype(int), 3)

print "\nSome samples..."
for i in range(20):
    print x_data_tr[i], " -> ", y_data_tr[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

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
