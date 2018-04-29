import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

#Draw example

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(trX,trY)

plt.grid()
plt.show()

#define palceholders and variables

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(0.0, name="weights") 

#define model, cost function and GradientDescentOptimizer

y_model = w*x
cost = tf.square(y- y_model)  

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess :   
  tf.global_variables_initializer().run()
  
  for i in range(100) :
    for (a,b) in zip(trX, trY) :
      sess.run(optimizer, feed_dict = {x: a, y:b})
    print(f"w: {sess.run(w)}  y_model: {sess.run(y_model, feed_dict = {x: a})}  cost: {sess.run(cost, feed_dict = {x: a, y: b})}")
    
