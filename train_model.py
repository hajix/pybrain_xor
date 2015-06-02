# -*- coding: utf-8 -*-

import pickle
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer  # ,TanhLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


def train_model():
    # build your network
    net = FeedForwardNetwork()

    # make layers of network
    # activation functions : (Linear|Sigmoid|TanhLayer)Layer
    inLayer = SigmoidLayer(2)
    hiddenLayer1 = SigmoidLayer(4)
    hiddenLayer2 = SigmoidLayer(4)
    outLayer = LinearLayer(1)

    # add those layers to network
    net.addInputModule(inLayer)
    net.addModule(hiddenLayer1)
    net.addModule(hiddenLayer2)
    net.addOutputModule(outLayer)

    # set connection between layers
    in_to_hidden = FullConnection(inLayer, hiddenLayer1)
    hidden_to_hidden = FullConnection(hiddenLayer1, hiddenLayer2)
    hidden_to_out = FullConnection(hiddenLayer2, outLayer)

    # add connections to the network
    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_hidden)
    net.addConnection(hidden_to_out)

    # final step to make your network usable
    net.sortModules()

    # make data set
    # ds1 -> unsigned
    ds1 = SupervisedDataSet(2, 1)
    ds1.addSample((0, 0), (0,))
    ds1.addSample((0, 1), (1,))
    ds1.addSample((1, 0), (1,))
    ds1.addSample((1, 1), (0,))
    # ds2 -> signed
    ds2 = SupervisedDataSet(2, 1)
    ds2.addSample((-1, -1), (-1,))
    ds2.addSample((-1, +1), (+1,))
    ds2.addSample((+1, -1), (+1,))
    ds2.addSample((+1, +1), (-1,))

    # make trainer based on network
    trainer = BackpropTrainer(net, dataset=ds2, learningrate=0.003, momentum=0.99, verbose=False)

    # train the trainer
    trainer.trainEpochs(1500)

    # return model
    return net


if __name__ == '__main__':
    # pickle my model
    try:
        f = open('_learned.mdl', 'r')
        net = pickle.load(f)
        f.close()
        print ('model was trained previously')
        ch = raw_input('press r to train new model or any key to exit\n')
        if (ch == 'r'):
            print ('training new model...')
            raise ValueError('you decide to train new model...')
        print ('you decide to exit...')
    except:
        net = train_model()
        f = open('_learned.mdl', 'w')
        pickle.dump(net, f)
        f.close()
        print ('new model trained...')

