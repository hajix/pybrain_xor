# -*- coding: utf-8 -*-

import pickle

if __name__ == '__main__':
    try:
        f = open('_learned.mdl', 'r')
        net = pickle.load(f)
        f.close()
        print ('model loaded successfully')
        # test the network
        print (('test with unsigned values'))
        print ((net.activate([0, 0])))
        print ((net.activate([0, 1])))
        print ((net.activate([1, 0])))
        print ((net.activate([1, 1])))
        print (('test with signed values'))
        print ((net.activate([-1, -1])))
        print ((net.activate([-1, +1])))
        print ((net.activate([+1, -1])))
        print ((net.activate([+1, +1])))
    except:
        print ('ni model found')
