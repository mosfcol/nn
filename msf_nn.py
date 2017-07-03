# coding=utf-8

import random
import numpy as np

# helpers
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(mat):
    assertThat(mat.ndim == 1)
    if mat.ndim == 1:
        maxElement = np.amax(mat)
        mat = mat - maxElement
        elementWiseExp = np.exp(mat)
        return elementWiseExp / np.sum(elementWiseExp)
    else:
        rowWiseMax = np.max(mat, axis = 1).reshape((-1, 1))
        mat -= rowWiseMax
        elementWiseExp = np.exp(mat)
        rowWiseSum = np.sum(elementWiseExp, axis = 1).reshape((-1, 1))
        return elementWiseExp / rowWiseSum

class NeuralNet:
    """
    it should have 2 Ws and 2bs
    the params are trainable
    """
    def __init__(self, Dx, Dh, Dy):
        self.Dx = Dx
        self.Dh = Dh
        self.Dy = Dy
        self.W1 = 0.5 * np.random.randn(Dx, Dh)
        self.b1 = 0.5 * np.random.randn(1, Dh) # already with (1, Dh) shape
        self.W2 = 0.5 * np.random.randn(Dx, Dy)
        self.b2 = 0.5 * np.random.randn(1, Dy) # already with (1, Dy) shape

    def train(self, x, y):
        print "training"
        assertThat(y.shape == (self.Dy,))
        assertThat(x.shape == (self.Dx,))
        a1 = np.atleast_2d(x)
        z1 = np.atleast_2d(x)
        assertThat(z1.shape == (1, self.Dx))
        z2 = np.dot(a1, self.W1) + self.b1
        assertThat(z2.shape == (1, self.Dh))

        a2 = sigmoid(z2)
        assertThat(a2.shape == (1, self.Dh))

        z3 = np.dot(a2, self.W2) + self.b2
        assertThat(z3.shape == (1, self.Dy))

        #a3 = [ softmax(z3[0]) ]
        a3 = np.array([ softmax(z3[0]) ])
        assertThat(a3.shape == (1, self.Dy))

        J = CE(y, a3[0]) # y, y_predict
        print "J = " + str(J)

        theta3 = a3 - np.atleast_2d(y) # numeric checked
        assertThat(theta3.shape == (1, self.Dy))

        theta3_numeric = numeric_gradient(lambda input: CE(y, softmax(input)), z3[0])
        print "theta3_numeric = " + str(theta3_numeric)

        deltaW2 = a2.T.dot(theta3) # numeric checked
        print "deltaW2 = " + str(deltaW2)
        #gradient check
        j_given_w2 = lambda w2 : CE(y, softmax((a2.dot(w2) + self.b2)[0]))
        deltaW2_numeric = numeric_gradient(j_given_w2, self.W2)
        print "deltaW2_numeric = " + str(deltaW2_numeric)

        deltaB2 = theta3  # numeric checked
        print "deltaB2 = " + str(deltaB2)
        #gradient check
        j_given_b2 = lambda b2 : CE(y, softmax((a2.dot(self.W2) + b2)[0]))
        deltaB2_numeric = numeric_gradient(j_given_b2, self.b2)
        print "deltaB2_numeric = " + str(deltaB2_numeric)

        ones_a2_shape = np.ones(a2.shape)
        grad_sigmoid_z2 = np.multiply(a2, (ones_a2_shape - a2))

        thetaA2 = theta3.dot(self.W2.T)
        assertThat(thetaA2.shape == (1, self.Dh))

        theta2 = np.multiply(grad_sigmoid_z2, thetaA2)
        assertThat(theta2.shape == (1, self.Dh))

        deltaW1 = a1.T.dot(theta2) # numeric checked
        assertThat(deltaW1.shape == self.W1.shape)
        print "deltaW1 = " + str(deltaW1)
        # gradient check
        j_given_w1_fixed_y = lambda w1 : self.j_given_w1(w1, a1, y)
        deltaW1_numeric = numeric_gradient(j_given_w1_fixed_y, self.W1)
        print "deltaW1_numeric = " + str(deltaW1_numeric)

        deltaB1 = theta2 #numeric checked
        print "deltaB1 = " + str(deltaB1)
        #gradient check
        j_given_b1_fixed_y = lambda b1 : self.j_given_b1(b1, a1, y)
        deltaB1_numeric = numeric_gradient(j_given_b1_fixed_y, self.b1)
        print "deltaB1_numeric = " + str(deltaB1_numeric)

        ALPHA = 0.02
        self.W1 = self.W1 - ALPHA * deltaW1
        self.W2 = self.W2 - ALPHA * deltaW2
        self.b1 = self.b1 - ALPHA * deltaB1
        self.b2 = self.b2 - ALPHA * deltaB2

    def  j_given_w1(self, w1, a1, y):
        z2 = a1.dot(w1) + self.b1
        a2 = sigmoid(z2)
        z3 = a2.dot(self.W2) + self.b2
        a3_ = softmax(z3[0])
        return CE(y, a3_)

    def j_given_b1(self, b1, a1, y):
        z2 = a1.dot(self.W1) + b1
        a2 = sigmoid(z2)
        z3 = a2.dot(self.W2) + self.b2
        a3_ = softmax(z3[0])
        return CE(y, a3_)

    def predict(self, x):
        a1 = np.atleast_2d(x)
        z2 = a1.dot(self.W1) + self.b1
        a2 = sigmoid(z2)
        z3 = a2.dot(self.W2) + self.b2
        a3 = softmax(z3[0])
        return a3

def assertThat(condition):
    if not condition:
        raise AssertionError()

def CE(y, y_predict):
    assertThat(y.ndim == 1)
    assertThat(y_predict.ndim == 1)
    return -np.sum(np.multiply(y, np.log(y_predict)))


def numeric_gradient(f, x):
    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh = f(x)
        x[ix] = old_value
        grad[ix] = (fxh - fx) / h
        it.iternext()
    return grad

def training():
    print "start training"
    nn = NeuralNet(2, 2, 2)
    print(nn.W1)
    print(nn.W2)
    print(nn.b1)
    print(nn.b2)

    # XOR
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    Y = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]])
    for i in range(200):
        dataIndex = i % 4
        print "dataIndex = " + str(dataIndex)
        nn.train(X[dataIndex], Y[dataIndex])

    print "PREDICT:"
    print(nn.predict(np.array([0., 0.])))
    print(nn.predict(np.array([0., 1.])))
    print(nn.predict(np.array([1., 0.])))
    print(nn.predict(np.array([1., 1.])))

def playground():
    z3ToEvaluate = np.array([1., 2., 3.])
    yToEvaluate = np.array([1., 0., 0.])
    grad = numeric_gradient(lambda z3 : J_given_z3(z3, yToEvaluate), z3ToEvaluate)
    print "numeric grad = " + str(grad)
    actual_grad = softmax(z3ToEvaluate) - yToEvaluate
    print "actual grad = " + str(actual_grad)

def J_given_z3(z3, y):
    return CE(y, softmax(z3))

def power2(x):
    return np.sum(x ** 2)

if __name__ == "__main__":
    training()
    #playground()