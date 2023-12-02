import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient

class MultilayerPreceptron:
    def __init__(self, data, labels, layers, normalize_data=False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed
        self.labels = labels
        self.layers = layers
        self.normalize_data = normalize_data
        self.thetas = MultilayerPreceptron.thetas_init(layers)

    def predict(self, data):
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]

        predictions = MultilayerPreceptron.feedforward_propagation(data_processed, self.thetas, self.layers)

        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    def train(self, max_iterations=1000, alpha=0.1):
        unrolled_theta = MultilayerPreceptron.thetas_unroll(self.thetas)
        (optimized_theta, cost_history) = MultilayerPreceptron.gradient_descent(self.data, self.labels, unrolled_theta, self.layers, max_iterations, alpha)
        self.thetas = MultilayerPreceptron.theta_roll(optimized_theta, self.layers)
        return self.thetas, cost_history

    @staticmethod
    def gradient_descent(data, labels, unrolled_theta, layers, max_iterations, alpha):
        optimized_theta = unrolled_theta
        cost_history = []

        for _ in range(max_iterations):
            cost = MultilayerPreceptron.cost_function(data, labels, MultilayerPreceptron.theta_roll(optimized_theta, layers), layers)
            cost_history.append(cost)
            theta_gradient = MultilayerPreceptron.gradient_step(data, labels, optimized_theta, layers)
            optimized_theta = optimized_theta - alpha*theta_gradient
        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        theta = MultilayerPreceptron.theta_roll((optimized_theta, layers))
        thetas_rolled_gradients = MultilayerPreceptron.back_propagation(data, labels, theta, layers)
        thetas_unrolled_gradients = MultilayerPreceptron.thetas_roll(thetas_rolled_gradients)
        return thetas_unrolled_gradients

    @staticmethod
    def back_propagation(data, labels, layers, theta):
        num_layers = len(layers)
        (num_examples, num_features) = data.shape
        num_label_types = layers[-1]

        deltas = {}

        for layer_index in range(num_layers -1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            deltas[layer_index] = np.zeros((out_count, in_count+1))
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            layers_activation = data[example_index,:].reshape((num_features, 1))
            layers_activations[0] = layers_activation

            for layers_index in range(num_layers -1):
                layers_theta = theta[layers_index]
                layer_input = np.dot(layers_theta,layers_activation)
                layers_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))
                layers_inputs[layers_index+1] = layer_input
                layers_activations[layers_index+1] = layers_activation
            output_layer_activation = layers_activations[1:, :]

            delta = {}
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]]=1
            #计算输出层与真实值的差异
            delta[num_layers -1] =output_layer_activation -1

            for layers_index in range(num_layers-2, 0, -1):
                layers_theta = theta[layer_index]
                next_delta = delta[layer_index+1]
                layer_input = layers_inputs[layers_index]
                layer_input = np.vstack(np.array((1), layer_input))
                delta[layer_index] = np.dot(layers_theta.T, next_delta)*sigmoid_gradient(layer_input)

                delta[layer_index] = delta[layer_index][1:,:]
            for layer_index in range(num_layers-1):
                layer_delta = np.dot(delta[layer_index+1], layers_activation[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta

        for layer_index in range(num_layers -1):
            deltas[layer_index] = delta[layer_index]*(1/num_examples)
        return deltas

    @staticmethod
    def cost_function(data, labels, thetas, layers):
        num_example = data.shape[0]
        num_labels = layers[-1]

        predictions = MultilayerPreceptron.feedforward_propagation(data, thetas, layers)
        bitwise_labels = np.zeros((num_example, num_labels))
        for example_index in range(num_example):
            bitwise_labels[example_index][labels[example_index][0]] = 1

        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_cost = np.sum(np.log(1-predictions[bitwise_labels == 0]))
        cost = (-1/num_example) * (bit_set_cost+bit_not_set_cost)
        return cost


    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        num_layers = len(layers)
        num_example = data.shape[0]
        in_layer_activation = data

        #逐层计算
        for layer_index in range(num_layers -1):
            theta = thetas[layer_index]
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))
            #考虑偏置项
            out_layer_activation = np.hstack((np.ones(num_example, 1)), out_layer_activation)
            in_layer_activation = out_layer_activation
        #不要偏置项
        return in_layer_activation[:, 1:]


    @staticmethod
    def theta_roll(unrolled_thetas, layers):
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]

            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_height + thetas_width
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layers_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layers_theta_unrolled.reshape((thetas_height, thetas_width))
            unrolled_shift = unrolled_shift + thetas_volume

        return thetas


    @staticmethod
    def thetas_unroll(thetas):
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([])
        for theta_layer_index in range(num_theta_layers):
            unrolled_theta = np.hstack(unrolled_theta, thetas[theta_layer_index].flatten())
        return unrolled_theta



    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)
        thetas = {}
        for layer_index in range(num_layers - 1):
            '''
            会执行两次，得到两组参数矩阵：25*785, 10*26
            '''
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            # 需要考虑到偏置项，偏置的个数跟输出的结果是一致的
            thetas[layer_index] = np.random.rand(out_count, in_count + 1)*0.05 #随机初始化，值尽量小点
        return thetas

