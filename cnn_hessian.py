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

def Gv(cost,s_,params,v,coefficient):
    # In theano multiplication of two vectors is a dot product and the input given by neural network to these is a list/tuple or a vector
    # So even if T.sum is removed from all this should be fine really.
    # Left multiply
    output = s_
    #Jv = T.Rop(output,params,v)
    JV = [T.grad(out,params)*v for out in output]
    Jv = T.prod(Jvi)
    # There is a missing T.sum which I feel is not required. Add later if error 
    HJv = T.grad(T.sum(T.grad(cost,output)*Jv), output, consider_constant=[Jv],disconnected_inputs='ignore')
    Gv = T.grad(T.sum(HJV*output),params,consider_constant=[HJv,Jv], disconnected_inputs='ignore')
    # Tikhonov Damping
    Gv = [T.as_tensor_variable(a) + coefficient*b for a,b in zip(Gv,v)]
    return Gv	

class training_cnn():

	def __init__(self,learning_rate = 0.1, n_epochs=1, dataset='mnist.pkl.gz',nkerns = [20,50], batch_size = 500 , testing =0):

		self.data = load_data(dataset)

	def cg(self,index,max_iter = 300):
	    # cg_last_x,v,coefficient 
	    #[b,cost_,s_] = train_model_1(index) - Theano doesn't support Multiple tensor output yet(http://stackoverflow.com/questions/27064617/theano-multiple-tensors-as-output)

	    cost_ = self.get_cost(index)
	    b = -self.get_grad(index)
	    s_ = self.get_s(index)

	    x = self.params
	    # cost,output,params,v,coefficient 
	    r = b - self.function_Gv(index)
	    d = r
	    delta_new = numpy.dot(r, r)
	    phi = []

	    for i in range (1,max_iter):

		    Ad = self.function_Gv(index)
		    alpha = delta_new/T.dot(d,Ad)
		    x = x + alpha*d

		    #Update the parameters before calculating the cost
		    #self.update_parameters(x)
		    # Update is not required as updating the assigned variable automatically updates the original variable too. 
		    #[b,cost_,s_] = train_model_1(index)

		    cost_ = self.get_cost(index)
		    b = -self.get_grad(index)
		    s_ = self.get_s(index)
		    if i%50==0:
		    	r = r - alpha*Ad
		    else: 
		    	r = b - self.function_Gv(index)

		    delta_old = delta_new
		    delta_new = T.dot(r,r)
		    beta = delta_new/delta_old
		    d = r+ beta*d

		    if i%20==0:
		    	validation_losses = [self.validate_model(i) for i in range(n_valid_batches)]
		    	this_validation_loss = numpy.mean(validation_losses)
	  		if this_validation_loss < self.best_validation_loss:
	  			self.best_validation_loss = this_validation_loss
	  			test_losses = [self.test_model(i) for i in range(n_test_batches)]
	  			test_score = numpy.mean(test_losses)
	  			print ('minibatch %i/%i, testing error %f %%' %(minibatch_index+1,n_train_batches,test_score*100.))

		    phi_i = -0.5 * numpy.dot(x, r + b)
		    phi.append(phi_i)

		    k = max(10, i/10)
		    if i > k and phi_i < 0 and (phi_i - phi[-k-1]) / phi_i < k*0.0005:
      			break


	def evaluate_lenet5(self,learning_rate = 0.1, n_epochs=1, dataset='mnist.pkl.gz',nkerns = [20,50], batch_size = 500 , testing =0):

		rng = numpy.random.RandomState(32324)

		datasets = self.data
		
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

		self.cost = layer_3.neg_log_likelihood(y)
	  	self.s = layer_3.s

	 	self.test_model = theano.function([index],layer_3.errors(y),
					givens={
							x: test_set_x[index*batch_size:(index+1)*batch_size],
							y: test_set_y[index*batch_size:(index+1)*batch_size]
							})

		self.validate_model = theano.function([index],layer_3.errors(y),
			givens={
					x: valid_set_x[index*batch_size:(index+1)*batch_size],
					y: valid_set_y[index*batch_size:(index+1)*batch_size]
					})

		self.train_predic = theano.function([index], layer_3.prob_y_given_x(),
				givens={
					x: train_set_x[index*batch_size:(index+1)*batch_size]
				})
		# list of parameters

		self.params = layer_3.params + layer_2.params + layer_1.params + layer_0.params
		grads = T.grad(self.cost,self.params)

		self.coefficient = 1 

		self.shapes = [i.get_value().shape for i in self.params]
		symbolic_types = T.scalar, T.vector, T.matrix, T.tensor3, T.tensor4
		v = [symbolic_types[len(i)]() for i in self.shapes]

		#import pdb	
    		#pdb.set_trace()
    		gauss_vector = Gv(self.cost,self.s,self.params,v,self.coefficient)
    		self.get_cost = theano.function([index,],self.cost,
			givens={
					x: train_set_x[index*batch_size:(index+1)*batch_size],
					y: train_set_y[index*batch_size:(index+1)*batch_size]
					}, on_unused_input='ignore')

    		self.get_grad = theano.function([index,],grads,
			givens={
					x: train_set_x[index*batch_size:(index+1)*batch_size],
					y: train_set_y[index*batch_size:(index+1)*batch_size]
					}, on_unused_input='ignore')

    		self.get_s = theano.function([index,],self.s,
			givens={
					x: train_set_x[index*batch_size:(index+1)*batch_size],
					}, on_unused_input='ignore')
    		self.function_Gv = theano.function([index],gauss_vector,givens={
					x: train_set_x[index*batch_size:(index+1)*batch_size],
					y: valid_set_y[index*batch_size:(index+1)*batch_size]
					},on_unused_input='ignore')
		# # Using stochastic gradient updates
		# updates = [ (param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip(params,grads) ]
		# train_model = theano.function([index],cost, updates=updates,
		# 	givens={
		# 			x: train_set_x[index*batch_size:(index+1)*batch_size],
		# 			y: train_set_y[index*batch_size:(index+1)*batch_size]
		# 			})

		# Using conjugate gradient updates
		# 'cg_ = cg(cost,output,params,coefficient,v)
				# updated_params = [(param_i, param_j) for param_i,param_j in zip(params,cg_)]'

		#self.update_parameters= theano.function([updated_params],updates=[params,updated_params])

		# -----------------------------------------Starting Training ------------------------------
		if testing ==0:
			print ('..... Training ' )

		# for early stopping
		patience = 10000
		patience_increase = 2
		improvement_threshold = 0.95
		validation_frequency = min(n_train_batches, patience//2)

		self.best_validation_loss = numpy.inf  # initialising loss to be inifinite
		best_itr = 0
		test_score = 0
		start_time = timeit.default_timer()

		epoch = 0
		done_looping = False
		
	  	

		while (epoch<n_epochs):
			epoch = epoch+1
			for minibatch_index in range(n_train_batches):
				iter = (epoch-1)*n_train_batches+minibatch_index

				if iter%1 ==0:
					print ('training @ iter = ', iter)

				self.cg(minibatch_index)
		if testing ==0 :
	  		print ('Optimization complete')
	  		print ('Best validation score of %f %% obtained at iteration %i,' 
	    			'with test performance %f %%' % (best_validation_loss*100., best_itr, test_score*100 ))
	  		print('The code ran for %.2fm' %((end_time - start_time)/60.))



	'''
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

	  					p_y_given_x =  [train_predic(i) for i in range(n_train_batches)]
	  					with open ('prob_best_model.pkl','wb') as f1:
	  						pickle.dump(p_y_given_x,f1)

	  			# if patience <= iter:
	  			# 	done_looping = True
	  			# 	break


	  	end_time = timeit.default_timer()
	  	# p_y_given_x =  [train_model(i) for i in range(n_train_batches)]
	  	# with open ('prob_best_model.pkl') as f:
	  	# 	pickle.dump(p_y_given_x)
	''' 

if __name__ == '__main__':
	training_cnn().evaluate_lenet5()
