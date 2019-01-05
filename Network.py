import random
import math
import statistics


def sigmoid(x):
    return 1/(1 + math.e ** (-x))


def sigmoidprime(x):
    return (math.e**x/(math.e**x+1)**2)


def dot(x, y):
    return [a*b for a, b in zip(x, y)]


class Network:  # The Actual Neural Network
    def __init__(self):
        self.weights1 = [[(random.random()-0.5)*2 for j in range(3)] for i in range(3)]
        self.weights2 = [(random.random()-0.5)*2 for i in range(3)]
        self.biases1 = [(random.random()-0.5)*2 for i in range(3)]
        self.biases2 = (random.random()-0.5)*2
        """print(self.weights1)
        print(self.weights2)
        print(self.biases1)
        print(self.biases2)"""

    def forwards(self, x):
        self.input = [int(i) for i in x]
        self.layer1r = [sum(dot(i, self.input)) + j for i, j in zip(self.weights1, self.biases1)]
        self.layer1 = [sigmoid(i) for i in self.layer1r]
        self.layer2r = [sum(dot(self.weights2, self.layer1)) + self.biases2]
        self.layer2 = [sigmoid(i) for i in self.layer2r]
        self.output = self.layer2[0]
        """print(self.input)
        print(self.layer1r)
        print(self.layer1)
        print(self.layer2r)
        print(self.layer2)
        print(self.output)"""
        return self.output

    def backwards(self, y):
        self.expected = int(y)
        deltal2 = [i*sigmoidprime(self.layer2r[0])*2*(self.expected - self.output) for i in self.weights2]
        self.deltaw1 = [[i*sigmoidprime(k)*j for j, k in zip(self.input, self.layer1r)] for i in deltal2]
        self.deltab1 = [i*sigmoidprime(j) for i, j in zip(deltal2, self.layer1r)]
        self.deltaw2 = [i*sigmoidprime(self.layer2r[0])*2*(self.expected - self.output) for i in self.layer1]
        self.deltab2 = sigmoidprime(self.layer2r[0])*2*(self.expected - self.output)
        """print(self.expected)
        print(self.deltaw1)
        print(self.deltab1)
        print(self.deltaw2)
        print(self.deltab2)"""

    def train(self, x, y):
        deltaw1 = []
        deltab1 = []
        deltaw2 = []
        deltab2 = []
        for i, j in zip(x, y):
            self.forwards(i)
            self.backwards(j)
            deltaw1.append(self.deltaw1)
            deltab1.append(self.deltab1)
            deltaw2.append(self.deltaw2)
            deltab2.append(self.deltab2)
        self.deltaw1 = [(list(map(statistics.mean, zip(*i)))) for i in list(map(list, zip(*deltaw1)))]
        self.deltab1 = (list(map(statistics.mean, zip(*deltab1))))
        self.deltaw2 = (list(map(statistics.mean, zip(*deltaw2))))
        self.deltab2 = statistics.mean(deltab2)
        self.weights1 = [[k + l for k, l in zip(i, j)] for i, j in zip(self.deltaw1, self.weights1)]
        self.biases1 = [i + j for i, j in zip(self.deltab1, self.biases1)]
        self.weights2 = [i + j for i, j in zip(self.weights2, self.deltaw2)]
        self.biases2 += self.deltab2

    def guess(self, x):
        print("Result: " + str(self.forwards(x)))
        return bool(round(self.forwards(x), 0))


# Learning Data
def randlist():
    return [[bool(round(random.random(), 0)) for i in range(3)] for j in range(100)]


def learn():  # The algorithm for learning, and checking all possible combinations at the end
    z = Network()
    data = randlist()
    data2 = [j[0] and j[1] or j[2] for j in data]
    for i in range(3000):
        z.train(data, data2)
    print(z.guess([False, False, False]))  # False
    print(z.guess([True, False, False]))  # False
    print(z.guess([False, True, False]))  # False
    print(z.guess([False, False, True]))  # True
    print(z.guess([False, True, True]))  # True
    print(z.guess([True, False, True]))  # True
    print(z.guess([True, True, False]))  # True
    print(z.guess([True, True, True]))  # True


learn()
