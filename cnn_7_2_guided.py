import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from logreg import load_data, LogReg	
import sys, numpy
from mlp import HiddenLayer
import timeit , pickle
#from cc import evaluate_lenet5 as guide


sys.path.insert(0, "/home/mayank/Downloads/computer_vision/Learning/")

class LeNetConvPoolLayer(object):
	def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2),border_mode = 'valid'):
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
		conv_out = conv2d(input=input,filters = self.W, filter_shape = filter_shape, input_shape = image_shape,border_mode = border_mode)

		# maxpooling

		pooled_out = downsample.max_pool_2d(input = conv_out,ds = poolsize, ignore_border = True, padding = (0,0) )

		self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))

		self.params = [self.W, self.b]

		self.input = input


def evaluate_lenet5(learning_rate = 0.10, n_epochs=200, dataset='mnist.pkl.gz',nkerns = [16,16,16,12,12,12], batch_size = 500):
	

	rng = numpy.random.RandomState(32324)

	datasets = load_data(dataset)
	
	train_set_x,train_set_y = datasets[0]
	valid_set_x,valid_set_y = datasets[1]
	test_set_x,test_set_y = datasets[2]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0]//batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]//batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]//batch_size

  	index = T.lscalar() # index for each mini batch
  	train_epoch = T.lscalar('train_epoch')

  	x = T.matrix('x')
  	y = T.ivector('y')

  	# ------------------------------- Building Model ----------------------------------
  	print "...Building the model"

  	
  	layer_0_input = x.reshape((batch_size,1,28,28))

  	# output image size = (28-5+1+)/1 = 24
  	layer_0 = LeNetConvPoolLayer(rng,input = layer_0_input, image_shape=(batch_size,1,28,28),
  		filter_shape=(nkerns[0],1,5,5),poolsize=(1,1))

  	#output image size = (24-3+1) = 22
  	layer_1 = LeNetConvPoolLayer(rng, input = layer_0.output, image_shape = (batch_size, nkerns[0],24,24), 
  								filter_shape = (nkerns[1],nkerns[0],3,3), poolsize=(1,1) )

  	#output image size = (22-3+1)/2 = 10
  	layer_2 = LeNetConvPoolLayer(rng, input = layer_1.output, image_shape = (batch_size, nkerns[1],22,22), 
  								filter_shape = (nkerns[2],nkerns[1],3,3), poolsize=(2,2) )

  	#output image size = (10-3+1)/2 = 4
  	layer_3 = LeNetConvPoolLayer(rng, input = layer_2.output, image_shape = (batch_size, nkerns[2],10,10),
  								filter_shape = (nkerns[3], nkerns[2],3,3), poolsize=(2,2) )

  	#output image size = (4-3+2+1) = 4
  	layer_4 = LeNetConvPoolLayer(rng, input = layer_3.output, image_shape = (batch_size, nkerns[3],4,4),
  								filter_shape = (nkerns[4], nkerns[3],3,3), poolsize=(1,1), border_mode = 1 )

  	#output image size = (4-3+1)/2 = 2
  	layer_5 = LeNetConvPoolLayer(rng, input = layer_4.output, image_shape = (batch_size, nkerns[4],4,4),
  								filter_shape = (nkerns[5], nkerns[4],3,3), poolsize=(2,2), border_mode = 1 )

  	# make the input to hidden layer 2 dimensional
  	layer_6_input = layer_5.output.flatten(2)

  	layer_6 = HiddenLayer(rng,input = layer_6_input, n_in = nkerns[5]*2*2, n_out = 200, activation = T.tanh)

  	layer_7 = LogReg(input = layer_6.output, n_in=200, n_out = 10)

  	teacher_p_y_given_x = theano.shared(numpy.asarray(pickle.load(open('prob_best_model.pkl','rb')),dtype =theano.config.floatX), borrow=True)
  	p_y_given_x = T.matrix('p_y_given_x')
  	e = theano.shared(value = 0, name = 'e', borrow = True)

  	cost = layer_7.neg_log_likelihood(y)  + 2.0/(e)*T.mean(-T.log(layer_7.p_y_given_x)*p_y_given_x - layer_7.p_y_given_x*T.log(p_y_given_x))
  	
	tg = theano.shared(numpy.asarray(pickle.load(open('modified_guided_data.pkl','rb')),dtype =theano.config.floatX), borrow=True)
  	guiding_weights = T.tensor4('guiding_weights')
        #guide_cost = T.mean(-T.log(layer_3.output)*guiding_weights - layer_3.output*T.log(guiding_weights))  
	guide_cost = T.mean((layer_3.output-guiding_weights)**2)
  	test_model = theano.function([index],layer_7.errors(y),
  				givens={
  						x: test_set_x[index*batch_size:(index+1)*batch_size],
  						y: test_set_y[index*batch_size:(index+1)*batch_size]
  						})

  	validate_model = theano.function([index],layer_7.errors(y),
			givens={
					x: valid_set_x[index*batch_size:(index+1)*batch_size],
					y: valid_set_y[index*batch_size:(index+1)*batch_size]
					})

  	# list of parameters

  	params = layer_7.params + layer_6.params + layer_5.params + layer_4.params + layer_3.params + layer_2.params + layer_1.params + layer_0.params
        params_gl = layer_3.params + layer_2.params + layer_1.params + layer_0.params
  	# import pdb
  	# pdb.set_trace()
        grads_gl = T.grad(guide_cost,params_gl)
        updates_gl = [ (param_i,param_i-learning_rate*grad_i) for param_i,grad_i in  zip(params_gl,grads_gl) ]
  	
  	grads = T.grad(cost,params)
        updates = [ (param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip(params,grads) ]

  	train_model = theano.function([index,train_epoch],cost, updates=updates,
			givens={
					x: train_set_x[index*batch_size:(index+1)*batch_size],
					y: train_set_y[index*batch_size:(index+1)*batch_size],
          			p_y_given_x: teacher_p_y_given_x[index],
          			e: train_epoch
					})
        train_till_guided_layer = theano.function([index],guide_cost,updates = updates_gl,
                        givens={
                                        x:  train_set_x[index*batch_size:(index+1)*batch_size],
                                        y:  train_set_y[index*batch_size:(index+1)*batch_size],
                                		guiding_weights : tg[index]
                                },on_unused_input='ignore')


  	# -----------------------------------------Starting Training ------------------------------

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
	while (epoch < n_epochs) and (not done_looping) :
  		epoch = epoch+1
  		for minibatch_index in range(n_train_batches):
  			iter = (epoch - 1)*n_train_batches + minibatch_index

  			if iter%100 ==0:
  				print ('training @ iter = ', iter)
			if epoch < n_epochs/5:
				cost_ij_guided = train_till_guided_layer(minibatch_index)
  			cost_ij = train_model(minibatch_index,epoch)
  			
			if(iter +1)%validation_frequency ==0:
  				# compute loss on validation set
  				validation_losses = [validate_model(i) for i in range(n_valid_batches)]
  				this_validation_loss = numpy.mean(validation_losses)

  				# import pdb
  				# pdb.set_trace()

            			with open('Student_6_terminal_out_2','a+') as f_:
  					f_.write('epoch %i, minibatch %i/%i, validation error %f %% \n' %(epoch,minibatch_index+1,n_train_batches,this_validation_loss*100. ))

  				# check with best validation score till now
  				if this_validation_loss<best_validation_loss:

  					# improve 
  					if this_validation_loss < best_validation_loss * improvement_threshold:
  						patience = max(patience, iter*patience_increase)

  					best_validation_loss = this_validation_loss
  					best_itr = iter

  					test_losses = [test_model(i) for i in range(n_test_batches)]
  					test_score = numpy.mean(test_losses)

            				with open('Student_6_terminal_out_2','a+') as f_:
  						f_.write('epoch %i, minibatch %i/%i, testing error %f %%\n' %(epoch, minibatch_index+1,n_train_batches,test_score*100.))
  					with open('best_model_7layer_2.pkl', 'wb') as f:
  						pickle.dump(params, f)
  					with open('Results_student_6_2.txt', 'wb') as f1:
  						f1.write(str(test_score*100)+'\n')
  			#if patience <= iter:
  			#	done_looping = True
  			#	break

  	end_time = timeit.default_timer()
	with open('Student_6_terminal_out_2','a+') as f_:
		f_.write('Optimization complete\n')
		f_.write ('Best validation score of %f %% obtained at iteration %i with test performance %f %% \n' % (best_validation_loss*100., best_itr, test_score*100 ))
		f_.write('The code ran for %.2fm\n' %((end_time - start_time)/60.))


'''	
        while (epoch < n_epochs) and (not done_looping) :
  		epoch = epoch+1
  		for minibatch_index in range(n_train_batches):
  			iter = (epoch - 1)*n_train_batches + minibatch_index

  			if iter%100 ==0:
  				print ('Guided training @ iter = ', iter)

  			cost_ij_guided = train_till_guided_layer(minibatch_index)
'''	                        
'''
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
  					if this_validation_loss < best_validation_loss * improvement_threshold:
  						patience = max(patience, iter*patience_increase)

  					best_validation_loss = this_validation_loss
  					best_itr = iter

  					test_losses = [test_model(i) for i in range(n_test_batches)]
  					test_score = numpy.mean(test_losses)

  					print ('epoch %i, minibatch %i/%i, testing error %f %%' %(epoch, minibatch_index+1,n_train_batches,test_score*100.))
                        '''
	#epoch = 0
if __name__ == '__main__':
	evaluate_lenet5()
