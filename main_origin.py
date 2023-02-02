from simple_neural_network import *


# Blog post example:

# nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
# for i in range(10000):
#    nn.train([0.05, 0.1], [0.01, 0.99])
#    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))


# XOR example:

training_sets = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

training_sets0 = list()
training_sets0.append(training_sets[0])


# training_sets = [
#     [[0, 0, 0], [0, 0]],
#     [[0, 1, 1], [0, 1]],
#     [[1, 0, 1], [1, 1]],
#     [[1, 1, 0], [0, 0]]
# ]


def main_origin():
    # for classic Xor example training_sets: num_inputs=2, num_outputs=1
    nn = NeuralNetwork(num_inputs=len(training_sets[0][0]), num_hidden=5, num_outputs=len(training_sets[0][1]))
    for i in range(10000):
        training_inputs, training_outputs = random.choice(training_sets)
        nn.train(training_inputs, training_outputs)

        # show sum of errors for rows of training set
        # print(i, nn.calculate_total_error(training_sets))
        # show error for zero row of training set
        print(i, nn.calculate_total_error(training_sets0))

    # recognize test patterns
    print(nn.calculate_total_error([[[0.1, 0.1], [0]]]))
    print(nn.calculate_total_error([[[0.1, 0.1], [1]]]))

    # print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

    # how to see just output of out neurons
    nn.feed_forward([0.9, 0.9])  # insert test pattern
    print(nn.output_layer.neurons[0].output)  # see result of recognition

    # nn.inspect()
    pass
