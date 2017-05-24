import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from logreg import load_data, LogReg	
import sys, numpy
from mlp import HiddenLayer
import timeit , pickle

sys.path.insert(0, "/home/mayank/Downloads/computer_vision/Learning/")

class LeNetConvPoolLayer(object):
	def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2)):
		'''
		image_shape - no. of images in a batch, no. of input features(layer m-1)	,
						image height, image width
		filter_shape - no. of filters or the no. of output features (layer m), no. of input features(layer m-1),
						filter height, filter width
		
		'''

		'''  
		For the case of input image(1st stage) it ensures that if the Image is RGB 
		then the filter depth is also 3 and if the image is B/W the filter depth is 1
		For further layers 
		'''
		
		assert image_shape[1]==filter_shape[1]
		self.input = input

		'''
		Input sise for each filter multiplication (of 2  matrices) 
		during convolution. 
		The no. of elements that would feed back the gradient to each one - so kind of like normalising
		'''
		fan_in = numpy.prod(filter_shape[1:])
		fan_out = numpy.prod(filter_shape[0]*filter_shape[2]*filter_shape[3])//numpy.prod(poolsize)

		'''
		Intialising weight and bias
		'''
		W_bound = numpy.sqrt(6./(fan_in+fan_out))
		self.W = theano.shared(numpy.asarray(rng.uniform(low = -W_bound, high = W_bound, size = filter_shape),dtype = theano.config.floatX)
				, borrow =True)
		self.b = theano.shared(numpy.zeros(shape = filter_shape[0], dtype = theano.config.floatX), borrow =True)

		'''
		convolving input and filters
		If I just write cone2d(input,..) instead of specifying by name that also works 
		'''
		conv_out = conv2d(input=input,filters = self.W, filter_shape = filter_shape, input_shape = image_shape)

		# maxpooling

		pooled_out = downsample.max_pool_2d(input = conv_out,ds = poolsize, ignore_border = True, padding = (0,0) )

		self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))

		self.params = [self.W, self.b]

		self.input = input


def evaluate_lenet5(learning_rate = 0.1, n_epochs=200, dataset='mnist.pkl.gz',nkerns = [20,50], batch_size = 500 , testing =0):
	

	rng = numpy.random.RandomState(32324)

	datasets = load_data(dataset)
	
	train_set_x,train_set_y = datasets[0]
	valid_set_x,valid_set_y = datasets[1]
	test_set_x,test_set_y = datasets[2]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0]//batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]//batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]//batch_size

  	index = T.lscalar() # index for each mini batch

  	x = T.matrix('x')
  	y = T.ivector('y')

  	# ------------------------------- Building Model ----------------------------------
  	if testing ==0:
  		print "...Building the model"

  	# output image size = (28-5+1)/2 = 12
  	layer_0_input = x.reshape((batch_size,1,28,28))
  	layer_0 = LeNetConvPoolLayer(rng,input = layer_0_input, image_shape=(batch_size,1,28,28),filter_shape=(nkerns[0],1,5,5),poolsize=(2,2))

  	#output image size = (12-5+1)/2 = 4
  	layer_1 = LeNetConvPoolLayer(rng, input = layer_0.output, image_shape = (batch_size, nkerns[0],12,12), 
  								filter_shape = (nkerns[1],nkerns[0],5,5), poolsize=(2,2) )

  	# make the input to hidden layer 2 dimensional
  	layer_2_input = layer_1.output.flatten(2)

  	layer_2 = HiddenLayer(rng,input = layer_2_input, n_in = nkerns[1]*4*4, n_out = 500, activation = T.tanh)

  	layer_3 = LogReg(input = layer_2.output, n_in=500, n_out = 10)

  	cost = layer_3.neg_log_likelihood(y)

   	test_model = theano.function([index],layer_3.errors(y),
  				givens={
  						x: test_set_x[index*batch_size:(index+1)*batch_size],
  						y: test_set_y[index*batch_size:(index+1)*batch_size]
  						})

  	validate_model = theano.function([index],layer_3.errors(y),
			givens={
					x: valid_set_x[index*batch_size:(index+1)*batch_size],
					y: valid_set_y[index*batch_size:(index+1)*batch_size]
					})

  	train_predic = theano.function([index], layer_3.prob_y_given_x(),
  			givens={
  				x: train_set_x[index*batch_size:(index+1)*batch_size]
  			})

  	# list of parameters
	layer_guided = theano.function([index], layer_1.output,
  			givens={
  				x: train_set_x[index*batch_size:(index+1)*batch_size]
  			})

  	params = layer_3.params + layer_2.params + layer_1.params + layer_0.params

  	grads = T.grad(cost,params)

  	updates = [ (param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip(params,grads) ]

  	train_model = theano.function([index],cost, updates=updates,
			givens={
					x: train_set_x[index*batch_size:(index+1)*batch_size],
					y: train_set_y[index*batch_size:(index+1)*batch_size]
					})

  	# -----------------------------------------Starting Training ------------------------------
  	if testing ==0:
  		print ('..... Training ' )

  	# for early stopping
  	patience = 10000
  	patience_increase = 2

  	improvement_threshold = 0.95

  	validation_frequency = min(n_train_batches, patience//2)

  	best_validation_loss = numpy.inf  # initialising loss to be inifinite
  	best_itr = 0
  	test_score = 0

  	start_time = timeit.default_timer()

  	epoch = 0
  	done_looping = False

  	while (epoch < n_epochs) and (not done_looping) and testing ==0:
  		epoch = epoch+1
  		for minibatch_index in range(n_train_batches):
  			iter = (epoch - 1)*n_train_batches + minibatch_index

  			if iter%100 ==0:
  				print ('training @ iter = ', iter)

  			cost_ij = train_model(minibatch_index)

  			if(iter +1)%validation_frequency ==0:
  				# compute loss on validation set
  				validation_losses = [validate_model(i) for i in range(n_valid_batches)]
  				this_validation_loss = numpy.mean(validation_losses)

  				# import pdb
  				# pdb.set_trace()
  				print ('epoch %i, minibatch %i/%i, validation error %f %%' %(epoch,minibatch_index+1,n_train_batches,this_validation_loss*100. ))

  				# check with best validation score till now
  				if this_validation_loss<best_validation_loss:

  					# improve 
  					# if this_validation_loss < best_validation_loss * improvement_threshold:
  					# 	patience = max(patience, iter*patience_increase)

  					best_validation_loss = this_validation_loss
  					best_itr = iter

  					test_losses = [test_model(i) for i in range(n_test_batches)]
  					test_score = numpy.mean(test_losses)

  					print ('epoch %i, minibatch %i/%i, testing error %f %%' %(epoch, minibatch_index+1,n_train_batches,test_score*100.))

  					with open('best_model.pkl', 'wb') as f:
  						pickle.dump(params, f)

  					with open('Results_teacher.txt','wb') as f2:
  						f2.write(str(test_score*100) + '\n')

  					p_y_given_x =  [train_predic(i) for i in range(n_train_batches)]
  					with open ('prob_best_model.pkl','wb') as f1:
  						pickle.dump(p_y_given_x,f1)

  			# if patience <= iter:
  			# 	done_looping = True
  			# 	break

	layer_2_op_dump = [layer_guided(i) for i in range(n_train_batches)]
	with open ('layer_guided.pkl','wb') as lg:
  		pickle.dump(layer_2_op_dump,lg)




  	end_time = timeit.default_timer()
  	# p_y_given_x =  [train_model(i) for i in range(n_train_batches)]
  	# with open ('prob_best_model.pkl') as f:
  	# 	pickle.dump(p_y_given_x)
  	
  	if testing ==0 :
  		print ('Optimization complete')
  		print ('Best validation score of %f %% obtained at iteration %i,' 
    			'with test performance %f %%' % (best_validation_loss*100., best_itr, test_score*100 ))
  		print('The code ran for %.2fm' %((end_time - start_time)/60.))

if __name__ == '__main__':
	evaluate_lenet5()

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


