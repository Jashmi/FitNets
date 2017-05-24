import theano
import numpy,pickle
from theano import tensor as T

tmp = numpy.asarray(pickle.load(open('layer_guided.pkl','rb')),dtype =theano.config.floatX)

tmp1 = numpy.empty((100,500,12,4,4))

for i in xrange(100):
    for j in xrange(500):
        for l in xrange(4):
            for m in xrange(4):
                for k in xrange(12):
                    tmp1[i][j][k][l][m] = max([tmp[i][j][k*4+q][l][m] for q in xrange(4)])


with open('modified_guided_data.pkl','wb') as f:
    pickle.dump(tmp1,f)
