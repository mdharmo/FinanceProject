import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data
from imblearn.over_sampling import SMOTE as smote
from sklearn.metrics import log_loss, f1_score

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def model_setup():

    num_conv_layers = 2

    num_filters = np.zeros(num_conv_layers).astype('int')
    filter_size = np.zeros(num_conv_layers).astype('int')

    for i in range(num_conv_layers):
        filter_size[i] = 5
        num_filters[i] = 64

    fc_size = 128
    num_channels = 1
    num_classes = 10
    img_size_flat = 784
    img_size = 28
    total_epochs = 50

    return num_filters,filter_size,fc_size,num_channels,num_classes,img_size_flat,img_size,total_epochs

def new_conv_layer(input,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,  # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.


    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features



def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def load_data():

    return input_data.read_data_sets('data/MNIST/', one_hot=True)

def train(data,session,optimizer,accuracy,train_batch_size,x,y_true):

    num_iterations = int(np.floor(60000/train_batch_size)+1)

    acc = 0
    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        acc += session.run(accuracy, feed_dict=feed_dict_train)/float(num_iterations)


    return acc

def predict(data,x,y_true,session,y_pred_cls,y_pred,cost):

    # Number of images in the test-set.
    num_test = int(len(data.images))

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.


    # I'm just doing it one at a time.
    feed_dict = {x: data.images[:],
                     y_true: data.labels[:]}
    true_cost = session.run(cost,feed_dict=feed_dict)
    pred = session.run(y_pred,feed_dict=feed_dict)
    cls_pred = session.run(y_pred_cls,feed_dict=feed_dict)

    return cls_pred,num_test,pred,true_cost

def build_model(num_filters,filter_size,fc_size,num_channels,num_classes,x,
                x_image,y_true,y_true_cls):


    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                       num_input_channels=num_channels,
                       filter_size=filter_size[0],
                       num_filters=num_filters[0],
                       use_pooling=True)

    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       num_input_channels=num_filters[0],
                       filter_size=filter_size[1],
                       num_filters=num_filters[1],
                       use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv2)

    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             use_relu=False)

    y_pred = tf.nn.softmax(layer_fc2)

    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                            labels=y_true)

    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return optimizer,accuracy,y_pred_cls,y_pred,cost


def calc_f1(pred_labels,true_labels):

    tempvalf1 = f1_score(y_true=np.argmax(true_labels,axis=-1),
                             y_pred=np.argmax(pred_labels,axis=-1), average=None)
    current_f1 = np.average(tempvalf1)

    cross_entropy_check = log_loss(true_labels,pred_labels)

    return current_f1, cross_entropy_check

def model_save_test(num_epochs,current_f1,best_f1,patience,saver,session):

    if current_f1>best_f1 + np.finfo(float).eps:
        best_f1 = current_f1
        patience=0
        #Save new Model
        saver.save(session, "/home/mharmon/FinanceProject/ModelResults/tftest/model" + str(num_epochs) + '.cpkt')
        print('New Model Saved')
    else:
        patience+=1


    return patience,best_f1

def restore_model(session,saver,num_epochs):

    saver.restore(session, "/home/mharmon/FinanceProject/ModelResults/tftest/model" + str(num_epochs) + '.cpkt')
    print("Model restored.")

    return
def main():


    num_filters, filter_size, fc_size, num_channels, num_classes,\
    img_size_flat, img_size, total_epochs = model_setup()

    # All of this should stuff should remain in the main() function...
    data = load_data()
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    data.test.cls = np.argmax(data.test.labels, axis=1)
    data.validation.cls = np.argmax(data.validation.labels,axis=1)

    print('Building Model')
    optimizer,accuracy,y_pred_cls,y_pred,cost = build_model(num_filters,filter_size,fc_size,num_channels,num_classes,x,
                x_image,y_true,y_true_cls)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train_batch_size = 64
    classes = 10
    current_f1 = 0.
    best_f1 = current_f1
    num_epochs = 0
    patience = 0
    print()
    print('Begin Training')
    print()
    while num_epochs <= total_epochs and patience<5:

        start_time = time.time()
        acc = train(data.train, session, optimizer, accuracy, train_batch_size, x, y_true)
        end_time = time.time()
        print('Training Epoch ' + str(num_epochs) + ' With Accuracy %.4f' %acc)
        # Difference between start and end-times.
        time_dif = end_time - start_time
        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

        # Predict on the validation data
        cls_pred,num_test,pred,val_cost = predict(data.validation,x,y_true,session,y_pred_cls,y_pred,cost)

        current_f1, cross_entropy_check = calc_f1(pred, data.validation.labels)

        patience,best_f1 = model_save_test(num_epochs, current_f1, best_f1, patience, saver,session)
        print()

        num_epochs+=1


    # Restore best model for testing:
    restore_model(session,saver,num_epochs-1)
    # Predict on the test data
    cls_pred,num_test,pred,cost = predict(data.test,x,y_true,session,y_pred_cls,y_pred,cost)
    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))


    return

if __name__=='__main__':
    main()