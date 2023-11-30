import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient

class MultilayerPreceptron:
    def __init__(self, data, labels,layers, normalize_data = False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)
        self.data = data_processed
        self.labels = labels
        self.layers = layers
        self.normalize_data = normalize_data
        self.thetas = MultilayerPreceptron.thetas_init(layers)
    def train(self,max_iterations=1000,alpha=0.1):
        unrolled_theta = MultilayerPreceptron.thetas_unroll(self.thetas)
        MultilayerPreceptron.gradient_descent(self.data, self.labels, unrolled_theta, self.layers, max_iterations, alpha)





    @staticmethod
    def gradient_descent(data,labels, unrolled_theta,layers, max_iterations, alpha):
        optimized_theta = unrolled_theta
        cost_history = []

        for _ in range(max_iterations):
            cost = MultilayerPreceptron.cost_function(data, labels, MultilayerPreceptron.theta_roll(unrolled_theta), layers)
            cost_history.append(cost)
            theta_gradient = MultilayerPreceptron.gradient_step(data, labels, optimized_theta, layers)

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        theta = MultilayerPreceptron.theta_roll((optimized_theta))
        MultilayerPreceptron.back_propagation()



    @staticmethod
    def back_propagation():
        




    @staticmethod
    def cost_function(data, labels, thetas, layers):
        num_layers = len(layers)
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

            thetas_width = in_count +1
            thetas_hight = out_count
            thetas_volume = thetas_hight + thetas_width
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layers_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layers_theta_unrolled.reshape((thetas_hight, thetas_width))
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
        for layers_index in range(num_layers -1):
            in_count = layers[layers_index]
            out_count = layers[layers_index+1]
            #需要考虑到偏置项，偏置的个数跟输出的结果是一致的
            thetas[layers_index] = np.random.rand(out_count, in_count + 1)*0.05 #随机初始化，值尽量小点
        return thetas

