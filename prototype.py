import pandas as pd
import numpy as np
import math
import random

#######################################################################

#load train and test set
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train_sample = train.fillna(value=0)

feature_names = list(train_sample.columns)
feature_names.remove("cam_seconds")
feature_names.remove("cam_time")
feature_names.remove("time")
feature_names.remove("HD_s")
feature_names.remove("SC_s")
feature_names.remove("SL_s")
feature_names.remove("EL_s")
feature_names.remove("WL_s")
feature_names.remove("HL_s")
feature_names.remove("SR_s")
feature_names.remove("ER_s")
feature_names.remove("WR_s")
feature_names.remove("HR_s")
feature_names.remove("truth5")
feature_names.remove("TRUTH")

# first try keep derived features
feature_names.remove("dist")
feature_names.remove("elaps")
feature_names.remove("trav")
feature_names.remove("speed")
feature_names.remove("wrist_dist")
feature_names.remove("elbow_dist")
feature_names.remove("LH_size")
feature_names.remove("RH_size")
feature_names.remove("LWS_dist")
feature_names.remove("RWS_dist")
feature_names.remove("width")
feature_names.remove("neck_len")
feature_names.remove("facing_angle")
feature_names.remove("LE_angle")
feature_names.remove("RE_angle")
feature_names.remove("LWoE")
feature_names.remove("RWoE")
feature_names.remove("LEoS")
feature_names.remove("REoS")
feature_names.remove("LWoS")
feature_names.remove("RWoS")
feature_names.remove("RWoLW")
feature_names.remove("REoLE")
feature_names.remove("RS_angle")
feature_names.remove("LS_angle")

features = train_sample[feature_names].values

# encode truth to numeric vector
# set(['dry', 'load', 'fetch_wip', 'unknown', 'inspect', 'box', 'walk', 'paint', 'toss', 'load_serial', 'fetch'])
# truth 'dry' would be [1,0,0,...,0], 'load' would be [0,1,0,...,0]
target_type=list(set(train_sample["TRUTH"].values))
target=np.zeros((train_sample.shape[0],len(target_type)))
for i in range(len(target)):
    for j in range(len(target_type)):
        if train_sample["TRUTH"].values[i]==target_type[j]:
            target[i][j]=1


#######################################################################

random.seed(0)
# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

# Back-Propagation Neural Networks
class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no
        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError, 'wrong number of inputs'
        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]
        # hidden activations
        for j in range(self.nh):
            summ = 0.0
            for i in range(self.ni):
                summ = summ + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(summ)
        # output activations
        for k in range(self.no):
            summ = 0.0
            for j in range(self.nh):
                summ = summ + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(summ)
        return self.ao[:]
    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError, 'wrong number of target values'
        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error
        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]
        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error
    def calculate_error(self, features, target):
        correct = 0
        for i in range(target.shape[0]):
            result = list(self.update(features[i]))
            if result.index(max(result)) == list(target[i]).index(max(target[i])):
                correct += 1
        print 'accuracy: ', correct/target.shape[0]
    def weights(self):
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]
    def train(self, features, target, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0
            for j in range(target.shape[0]):
                inputs = features[j]
                targets = target[j]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                pass #print 'error %-14f' % error
    def test(self, features, target):
        for i in range(target.shape[0]):
            print features[i], '->', self.update(features[i]),'-->', target[i]

def simpletest():
    # Teach network XOR function
    features = np.asarray([[0,0], [0,1], [1,0], [1,1]])
    target = np.asarray([[0], [1], [1], [0]])
    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(features, target)
    # test it
    n.test(features, target)
    n.calculate_error(features, target)
    
simpletest()

n=NN(55,2,11)
n.train(features, target)
# training error
n.calculate_error(features, target)

    

