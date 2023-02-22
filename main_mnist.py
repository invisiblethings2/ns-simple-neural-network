import tensorflow
from keras.datasets import mnist
import numpy
from pathlib import Path
from simple_neural_network import *
import time

from data import *
# import simple_neural_network as snn




###

def main_mnist2():

    data = Data()
    # 5 0 4 1 9..
    # itrain_list = itrain.tolist()
    # nn = NeuralNetwork(num_inputs=len(itrain_list[0]) * len(itrain_list[0][0]), num_hidden=784, num_outputs=10)
    nn = NeuralNetwork(num_inputs=len(data.itrain[0]), num_hidden=784, num_outputs=10,
                       hidden_layer_weights=data.hidden_layer_weights, hidden_layer_bias=data.hidden_layer_bias,
                       output_layer_weights=data.output_layer_weights, output_layer_bias=data.output_layer_bias
                       )

    # for i in range(100):
    #     flat_im = list(numpy.concatenate(itrain_list[i]).flat)
    #     flat_im = convert_to_fraction(flat_im)
    #     flat_label = convert_label_to_list(ltrain[i])
    #     flat_set.append([flat_im, flat_label])

    if not data.loaded:
        start_time = time.time()
        #for i in range(4000):
        for i in range(len(data.itrain)):
            nn.train(data.itrain[i], data.ltrain[i])
            if False:
                if i % 5 == 0:
                    print("{i}--------------")
                    test_idx = 0
                    number0 = data.itrain[test_idx]

                    print(f"wait for {convert_list_to_label(data.ltrain[test_idx])}={data.ltrain[test_idx]}")
                    nn.feed_forward(number0)
                    for i in range(len(nn.output_layer.neurons)):
                        print(f"{i}:{nn.output_layer.neurons[i].output}")

        data.hidden_layer_weights = nn.get_hidden_layer_weights()
        data.output_layer_weights = nn.get_output_layer_weights()
        data.hidden_layer_bias = nn.hidden_layer.bias
        data.output_layer_bias = nn.output_layer.bias
        data.save()
        elapsed_time = time.time() - start_time
        print(f'train: Elapsed time: {elapsed_time:.2f} seconds')
    else:
        #nn.inspect()
        pass

    #return
    if False:
        test_idx = 0
        number0 = data.itrain[test_idx]

        print(f"wait for {convert_list_to_label(data.ltrain[test_idx])}={data.ltrain[test_idx]}")
        nn.feed_forward(number0)
        for i in range(len(nn.output_layer.neurons)):
            print(f"{i}:{nn.output_layer.neurons[i].output}")
    elif False:
        for i in range(10):
            number = data.itest[i]
            print("--------------")
            print(f"wait for {convert_list_to_label(data.ltest[i])}={data.ltest[i]}")
            nn.feed_forward(number)
            for ii in range(len(nn.output_layer.neurons)):
                print(f"{ii}:{nn.output_layer.neurons[ii].output}")
    else:
        correct_recognitions = 0
        test_len = 10
        for i in range(test_len):
            number = data.itest[i]
            #start_time = time.time()
            nn.feed_forward(number)
            #print(f'test: Elapsed time: {time.time() - start_time:.2f} seconds')
            outs = list()
            for ii in range(len(nn.output_layer.neurons)):
                outs.append(nn.output_layer.neurons[ii].output)

            ind = numpy.argmax(outs)
            if ind == convert_list_to_label(data.ltest[i]):
                correct_recognitions += 1

        print(f"correct_recognitions:{correct_recognitions}")
        print(f"%{correct_recognitions*100/test_len}")

    pass


def main_mnist():
    # Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
    (itrain, ltrain), (itest, ltest) = mnist.load_data()

    print('X_train: ' + str(itrain.shape))
    print('Y_train: ' + str(ltrain.shape))
    print('X_test:  ' + str(itest.shape))
    print('Y_test:  ' + str(ltest.shape))

    # 5 0 4 1 9..
    itrain_list = itrain.tolist()
    nn = NeuralNetwork(num_inputs=len(itrain_list[0]) * len(itrain_list[0][0]), num_hidden=784, num_outputs=10)

    flat_set = list()
    for i in range(10):
        flat_im = list(numpy.concatenate(itrain_list[i]).flat)
        flat_im = convert_to_fraction(flat_im)
        flat_label = convert_label_to_list(ltrain[i])
        flat_set.append([flat_im, flat_label])

    for i in range(10):
        nn.train(flat_set[i][0], flat_set[i][1])

        # print(i, nn.calculate_total_error(flat_set))

    nn.inspect()

    test_idx = 0
    number0 = flat_set[test_idx][0]
    # for i in range(10):
    #    print(f"test {i} but we wait for 5 ----------------")
    #    nn.feed_forward(number0)
    # print(nn.output_layer.neurons[4].output)

    print(f"wait for {flat_set[test_idx][1]}")
    nn.feed_forward(number0)
    for i in range(len(nn.output_layer.neurons)):
        print(f"{i}:{nn.output_layer.neurons[i].output}")
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
