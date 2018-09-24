"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

def softmax(y_hat):
    e_x = np.exp(y_hat-np.max(y_hat))
    return e_x/np.sum(e_x,axis=1,keepdims=True)

def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    D,C = W.shape
    N = y.shape[0]
    
    for c in range(0,C):
        w_c = W[:,c]
        dw_c = np.zeros((D,))
        for i in range(0,N):
            x_i = X[i,:] #1xD
            likelihood = np.exp(np.inner(w_c.T,x_i))/np.sum(np.exp(np.inner(W.T,x_i)))#scalar
            if y[i]==c:
                loss -= np.log(likelihood)
                dw_c += x_i*(likelihood - 1)
            else:
                dw_c += x_i*likelihood
         
        dW[:,c] = dw_c/N
    
    loss = loss/N + reg*np.sum(np.square(W))/2
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW + reg*W
            


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    N = y.shape[0]
    _,C = W.shape
    
    y_hat = np.inner(X,W.T)
    likelihood = softmax(y_hat)
    
    log_likelihood = np.log(likelihood[range(N),y])
    
    loss = - np.sum(log_likelihood)/N + reg*np.sum(np.square(W))/2
    
    dW = likelihood
    dW[range(N),y] -= 1
    dW = np.inner(dW.T,X.T).T
    dW /= N
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW + reg*W

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val,num_iters=15000):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    #learning_rates = [1e-8, 1e-6]
    #regularization_strengths = [10e2, 10e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    learning_rates = np.linspace(1e-7,1e-5)
    regularization_strengths = np.linspace(1e2,1e5)

    for i in range(0,100):
        
        print("Test number ",i)
        
        lr = np.random.choice(learning_rates,1)[0]
        reg = np.random.choice(regularization_strengths,1)[0]
        
        softmax = SoftmaxClassifier()
        softmax.train(X_train, y_train, learning_rate=lr, reg=reg,num_iters=num_iters)
            
        y_train_pred = softmax.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_pred)
            
        y_val_pred = softmax.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)
            
        results[lr,reg] = (train_accuracy,val_accuracy)
            
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = softmax
        
        all_classifiers.append(softmax)
        
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
    
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
#    for (lr, reg) in sorted(results):
#        train_accuracy, val_accuracy = results[(lr, reg)]
#        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
#              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
