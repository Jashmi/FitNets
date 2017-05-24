
# TO enable usage of print as a function, which is not default yet but would be in future versions

from __future__ import print_function 

import six.moves.cPickle as pickle 
import gzip,os,sys,timeit,numpy,theano 
import theano.tensor as T 



class LogReg(object):


	def __init__(self,input,n_in,n_out): 


		self.W = theano.shared( value = numpy.zeros( (n_in,n_out), dtype = theano.config.floatX), name ='W', borrow = True) 
		self.b = theano.shared( value = numpy.zeros( (n_out,), dtype = theano.config.floatX), name = 'b', borrow=True) 
		
		self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b) 

		self.y_prediction = T.argmax(self.p_y_given_x, axis =1) 
		
		self.likelihood = T.max(self.p_y_given_x,axis=1) 
		# Keeping input and our parameters for future use of any definition
		self.params = [self.W,self.b] 
		
		self.input = input 

	def prob_y_given_x(self):
		return self.p_y_given_x

	def neg_log_likelihood(self,y):
		'''
		p_y_given_x is a matrix where each column corresponds to probs. of various classes
		for each input i . What we want to compute the -Log likelihood is only the probabilities
		for the correct class 
		'''

		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

	def errors(self,y):
		'''
		Calculate zero one loss over the function
		'''
		if y.ndim != self.y_prediction.ndim:
			raise TypeError('y should have the same shape as self.y_prediction','y',y.type,'self.y_prediction', self.y_prediction.type)

		return T.mean(T.neq(self.y_prediction,y))



def load_data(dataset):
		# dataset is the path to dataset mnist.pkl.gz
		with gzip.open(dataset,'rb') as f:
			train_set,valid_set,test_set = pickle.load(f)

		def shared_dataset(data_xy):
			data_x , data_y = data_xy
			shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX), borrow=True)
			shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX), borrow=True)
			return shared_x, T.cast(shared_y,'int32')
		
		test_set_x, test_set_y = shared_dataset(test_set)
		train_set_x, train_set_y = shared_dataset(train_set)
		valid_set_x, valid_set_y = shared_dataset(valid_set)

		return [(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)]

def sgd(learning_rate=0.13, n_epochs=200,dataset='mnist.pkl.gz',batch_size=600):
	dataset = load_data(dataset)
	
	train_set_x,train_set_y = dataset(0)
	valid_set_x,valid_set_y = dataset(1)
	test_set_x,test_set_y = dataset(2)

	n_train = train_set_x.get_value(borrow=True).shape[0] // batch_size  # '//' returns the floor integer
	n_valid = valid_set_x.get_value(borrow=True).shape[0] // batch_size
	n_test = test_set_x.get_value(borrow=True).shape[0] // batch_size

