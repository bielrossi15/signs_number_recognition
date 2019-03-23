import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def dataset():
	# loading train and test dataset and classes
	train_dataset = h5py.File('datasets/train_signs.h5', "r")
	train_set_features = np.array(train_dataset['train_set_x'][:]) # train features
	train_set_labels = np.array(train_dataset['train_set_y'][:]) # train labels

	test_dataset = h5py.File('datasets/test_signs.h5', "r")
	test_set_features = np.array(test_dataset['test_set_x'][:]) # test features
	test_set_labels = np.array(test_dataset['test_set_y'][:]) # test labels

	# classes = np.array(test_dataset['list_classes'][:]) # classes list

	depth = 6

	# converting labels to one hot
	train_set_labels = convert_to_one_hot(train_set_labels.reshape((1, train_set_labels.shape[0])), depth)
	test_set_labels = convert_to_one_hot(test_set_labels.reshape((1, test_set_labels.shape[0])), depth)
    
    
	# flatten train features (reducing dimensions)
	train_set_features_flat = train_set_features.reshape(train_set_features.shape[0], -1).T
	test_set_features_flat = test_set_features.reshape(test_set_features.shape[0], -1).T

	# normalizing train features 
	train_set_features = train_set_features_flat/255
	test_set_features = test_set_features_flat/255

	return train_set_features, train_set_labels, test_set_features, test_set_labels

def initializing_weights():
    W1 = tf.get_variable("W1", [25, 12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())
    

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_prop(X, parameters):

    w1 = parameters['W1']
    b1 = parameters['b1']
    w2 = parameters['W2']
    b2 = parameters['b2']
    w3 = parameters['W3']
    b3 = parameters['b3']
    # computing each layer
    first_layer = tf.nn.relu(tf.add(tf.matmul(w1, X), b1))
    second_layer = tf.nn.relu(tf.add(tf.matmul(w2, first_layer), b2))
    third_layer = tf.add(tf.matmul(w3, second_layer), b3)

    return third_layer


def model(iterations):

    x_train, y_train, x_test, y_test = dataset()
    x_size = x_train.shape[0]
    y_size = y_train.shape[0]
    costs = []
    learning_rate = 1e-4

    # placeholders
    X = tf.placeholder(tf.float32, shape=[x_size, None])
    Y = tf.placeholder(tf.float32, shape=[y_size, None])

    # weights
    parameters = initializing_weights()

    # forward prop
    third_layer = forward_prop(X, parameters)

    # back prop
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.transpose(Y), logits = tf.transpose(third_layer)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # run your session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # iterating over epochs without minibatches
        for epoch in range(iterations):
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})
            
            if epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)

        # calculating predictions
        correct_prediction = tf.equal(tf.argmax(third_layer), tf.argmax(Y))
		# accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        print('train accuracy:', accuracy.eval({X: x_train, Y: y_train}))
        print('test accuracy:', accuracy.eval({X: x_test, Y: y_test}))

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        return parameters


if __name__ == '__main__':
    iterations = int(input())
    parameters = model(iterations)
