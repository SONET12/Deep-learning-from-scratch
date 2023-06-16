
import tensorflow as tf



class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self,input_dim,output_dim):
        super(MyDenseLayer,self).__init__()

        #initialize weights and biases
        self.weights = self.add_weights((input_dim,output_dim))
        self.bias = self.add_weights((1,output_dim))

    def call (self, inputs):
        z= tf.matmul(inputs,self.weights)+self.bias

        output = tf.math.sigmoid(z)
        return output