import numpy as np

from exercise_code.layers import *
from exercise_code.layer_utils import *

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_params):
    """
    Convenience layer that performs an affine transform followed by a batch normalization and then by a ReLU
  
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
  
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    
    a, fc_cache = affine_forward(x, w, b)
    batch_out, batch_cache = batchnorm_forward(a, gamma, beta, bn_params)
    out, relu_cache = relu_forward(batch_out)
    cache = (fc_cache, batch_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batch_norm-relu convenience layer
    """

    fc_cache, batch_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dbatch, dgamma, dbeta = batchnorm_backward(da,batch_cache)
    dx, dw, db = affine_backward(dbatch, fc_cache)
    return dx, dw, db, dgamma, dbeta

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    
    The architecure should be affine - relu - affine - softmax.
  
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
  
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
    
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = np.random.normal(scale = weight_scale,size = (input_dim,hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale = weight_scale,size = (hidden_dim,num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
    
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
    
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        out_1,cache_1 = affine_relu_forward(X,W1,b1)
        out_2,cache_2 = affine_forward(out_1,W2,b2)
        scores = out_2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2))

        dx_2, grads['W2'], grads['b2'] = affine_backward(dscores, cache_2)
        dx_1, grads['W1'], grads['b1'] = affine_relu_backward(dx_2, cache_1)

        grads['W1'] += self.reg*W1
        grads['W2'] += self.reg*W2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(1,self.num_layers+1):
            self.params['W%d'%(i)] = np.random.normal(scale = weight_scale,size = (dims[i-1],dims[i]))
            self.params['b%d'%(i)] = np.zeros((1,dims[i]))
            if use_batchnorm and i!= self.num_layers:
                self.params['gamma%d'%(i)] = np.ones((1,dims[i]))
                self.params['beta%d'%(i)] = np.zeros((1,dims[i]))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            print(self.dropout_param)
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
    
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        #Store output values and cache from each layer in a list
        out = []
        cache = []
        
        #First layer
        if self.use_batchnorm:
            first_out,first_cache = affine_batchnorm_relu_forward(X, self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1'], self.bn_params[0])
        
        else:
            first_out,first_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        
        if self.use_dropout:
            first_out, dp_cache = dropout_forward(first_out, self.dropout_param)
            first_cache = (*first_cache,dp_cache)

        out.append(first_out)
        cache.append(first_cache)
               
        #Following layers excluding last one
        for i in range(1,self.num_layers-1):
            
            if self.use_batchnorm:
                aux_out,aux_cache = affine_batchnorm_relu_forward(out[-1], self.params['W%d'%(i+1)], self.params['b%d'%(i+1)], self.params['gamma%d'%(i+1)], self.params['beta%d'%(i+1)], self.bn_params[i])

            else:
                aux_out,aux_cache = affine_relu_forward(out[-1], self.params['W%d'%(i+1)], self.params['b%d'%(i+1)])

            if self.use_dropout:
                
                aux_out, dp_cache = dropout_forward(aux_out, self.dropout_param)
                aux_cache = (*aux_cache,dp_cache)

            out.append(aux_out)
            cache.append(aux_cache)
            
        #Last layer    
        last_out,last_cache = affine_forward(out[-1],self.params['W%d'%self.num_layers],self.params['b%d'%self.num_layers])
        out.append(last_out)
        cache.append(last_cache)
        scores = last_out
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5*self.reg*np.sum(self.params['W%d'%self.num_layers]**2)
        
        #last layer
        dx, grads['W%d'%self.num_layers],grads['b%d'%self.num_layers] = affine_backward(dscores,cache[-1])
        grads['W%d'%self.num_layers] += self.reg*self.params['W%d'%self.num_layers]
        
        reg_loss = 0
        
        #Previous layers
        
        for i in reversed(range(0,self.num_layers-1)):
            
            if self.use_dropout:
                dx = dropout_backward(dx,cache[i][-1])                
            
            if self.use_batchnorm:
                dx, grads['W%d'%(i+1)],grads['b%d'%(i+1)],grads['gamma%d'%(i+1)],grads['beta%d'%(i+1)] = affine_batchnorm_relu_backward(dx, cache[i][0:3])

            else:
                dx, grads['W%d'%(i+1)],grads['b%d'%(i+1)] = affine_relu_backward(dx, cache[i][0:2])        
            
            reg_loss += 0.5*self.reg*np.sum(self.params['W%d'%(i+1)]**2)
            grads['W%d'%(i+1)] += self.reg*self.params['W%d'%(i+1)] 
            
        loss+=reg_loss
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
