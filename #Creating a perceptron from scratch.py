#Creating a perceptron from scratch
'''
A neural network is just an advanced version of a perceptron 

Note:1) Will be using sigmoid activation function in the hidden neuron
2) The output will be given through logistic function 
3) There are 3 inputs so 3 neurons in the input layer
'''

import numpy as np
def sigmoid_function(x):
    function_value=1/(1+np.exp(-x))
    return function_value

def derivative_of_sigmoid(x):
    derivative= x*(1-x)
    return derivative

input_values=np.array([[0,0,1],
                      [1,1,1],
                      [1,0,1],
                      [0,1,1]])
output_values=np.array([[0],
                        [1],
                        [1],
                        [0]])

np.random.seed(1)
synaptic_weights= (2 * np.random.random((3,1)))-1

print("The weights assigned are:",synaptic_weights)

for iteration in range(100):
    input_layer= input_values
    prediction = sigmoid_function(np.dot(input_values,synaptic_weights))
    error= output_values - prediction
    tuning= error * derivative_of_sigmoid(prediction)
    synaptic_weights= synaptic_weights + np.dot(input_layer,tuning)
#Keeping the bias as 0
print("The prediction is",prediction)