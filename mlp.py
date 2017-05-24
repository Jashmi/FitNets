import theano,numpy
from theano import tensor as T

class HiddenLayer(object):
	def __init__(self,rng,input,n_in,n_out,activation=T.tanh):
		self.input = input

		W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(n_in+n_out)), high = numpy.sqrt(6./(n_in+n_out)), size=(n_in,n_out)), 
			dtype=theano.config.floatX )
		W = theano.shared(value = W_values, name='W',borrow=True)
		

		b_values= numpy.zeros(shape=(n_out,), dtype= theano.config.floatX)
		b = theano.shared(value=b_values, name='b', borrow=True)
		self.W = W
		self.b = b

		lin_output = T.dot(input,self.W)+self.b
		
		self.output = (lin_output if activation is None else activation(lin_output))
		
		self.params = [self.W , self.b]
