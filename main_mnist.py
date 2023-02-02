import tensorflow
from keras.datasets import mnist
import numpy


from simple_neural_network import *
#import simple_neural_network as snn


def convert_label_to_list(digit):
    label_list = [0] * 10
    label_list[digit] = 1
    return label_list


def make_flat_set(flat_im, flat_label):
    flat_set = list()
    for i in range(len(flat_im)):
        flat_set.append([flat_im[i], flat_label[i]])
    return flat_set


def convert_to_fraction(flat_im):
    res = list()
    koeff = 1 / 256
    for i in flat_im:
        res.append(flat_im[i] * koeff)
    return res


###

def main_mnist():
    (itrain, ltrain), (itest, ltest) = mnist.load_data()

    print('X_train: ' + str(itrain.shape))
    print('Y_train: ' + str(ltrain.shape))
    print('X_test:  ' + str(itest.shape))
    print('Y_test:  ' + str(ltest.shape))

    itrain_list = itrain.tolist()
    im1 = itrain_list[0]
    flat_im1 = list(numpy.concatenate(im1).flat)
    label1 = convert_label_to_list(ltrain[0])
    #nn = NeuralNetwork(num_inputs=len(itrain_list[0]) * len(itrain_list[0][0]), num_hidden=784, num_outputs=10, [0] * 784 * 784)
    nn = NeuralNetwork(num_inputs=len(itrain_list[0]) * len(itrain_list[0][0]), num_hidden=784, num_outputs=10)

    flat_set = list()
    for i in range(10):
        flat_im = list(numpy.concatenate(itrain_list[i]).flat)
        flat_im = convert_to_fraction(flat_im)
        flat_label = convert_label_to_list(ltrain[i])
        flat_set.append([flat_im, flat_label])

    for i in range(10):
        nn.train(flat_set[i][0], flat_set[i][1])

        #print(i, nn.calculate_total_error(flat_set))

    number0 = flat_set[i][0]
    nn.feed_forward(number0)
    #print(nn.output_layer.neurons[4].output)
    for i in range(len(nn.output_layer.neurons)):
        print(nn.output_layer.neurons[i].output)

    # for i in range(10000):
    #     flat_im = list(numpy.concatenate(itrain_list[i]).flat)
    #     flat_label = convert_label_to_list(ltrain[i])
    #     nn.train(flat_im, flat_label)
    #     # need [784.digitt], [label] array
    #
    #     flat_set = make_flat_set(flat_im, flat_label)
    #
    #     print(i, nn.calculate_total_error(flat_set))

    # print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

    pass
