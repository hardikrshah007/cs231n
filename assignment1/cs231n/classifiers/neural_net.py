from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).            
    scores_firstLayer = X.dot(W1) +b1
    H1 = np.maximum(0,scores_firstLayer)
    scores = H1.dot(W2) +b2

    #############################################################################
    # pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.
    for i in range(N):
    	scores_exp = np.exp(scores[i])
    	loss+= -np.log(np.exp(scores[i,y[i]])/np.sum(scores_exp))
    loss = loss/N + reg*np.sum(W1*W1) + reg*np.sum(W2*W2)
    #############################################################################
    # pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    grad_W2 = np.zeros_like(W2)
    grad_b2 = np.zeros_like(b2)

    scores_exp = np.exp(scores)
    sum_scores_exp = np.sum(scores_exp,axis=1)
    # correct_score =np.zeros(N)
    scores_exp_normalized = np.zeros_like(scores_exp)
    # correct_score = scores[range(N), y]

    for i in range(N):
    	scores_exp_normalized[i,:] = scores_exp[i,:]/sum_scores_exp[i]
    # input_data_transpose = H1.transpose()
    # grad_W2 = input_data_transpose.dot((scores_exp_normalized))
    # for i in range(N):
    # 	grad_W2[:,y[i]] += -H1[i]
    # grad_W2/=N
    # grad_W2+=2*reg*W2

    # grads['W2'] = grad_W2

    # for i in range(N):
    # 	grad_b2 += np.exp(H1[i,:].dot(W2)+b2)/sum_scores_exp[i]
    # 	grad_b2[y[i]]+=-1
    # grad_b2/=N

    # grads['b2'] = grad_b2


############ Layered Implementation##################
    
    dS = scores_exp_normalized
    for i in range(N):
    	dS[i,y[i]]+=-1
    grad_W2_layered = H1.transpose().dot(dS)

    grads['W2'] = grad_W2_layered/N +2*reg*W2
    grads['b2'] = np.sum(dS,axis=0)/N

    dH = dS.dot(W2.transpose())
    dscore_first_layer = np.multiply(scores_firstLayer>0,dH)
    db1 = np.sum(dscore_first_layer,axis=0)/N
    dW1 = X.transpose().dot(dscore_first_layer)
    grads['W1'] = dW1/N + 2*reg*W1
    grads['b1'] = db1

#####################################################
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """


    """
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      rand_num_list = np.random.choice(num_train,batch_size,replace=False)

      X_batch = None
      y_batch = None
      X_batch = X[rand_num_list]
      y_batch = y[rand_num_list]
      # print("The shape of X_batch is : ",X_batch.shape)
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      # pass
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      # pass
      self.W =self.W -grad*learning_rate 
    """
    num_train = X.shape[0]
    print("the num_train is:",num_train)

    iterations_per_epoch = 5*max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None
      rand_num_list = np.random.choice(num_train,batch_size)
      X_batch = None
      y_batch = None
      X_batch = X[rand_num_list]
      y_batch = y[rand_num_list]
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.     
      self.params['W1'] =self.params['W1'] -grads['W1']*learning_rate
      self.params['W2'] =self.params['W2'] -grads['W2']*learning_rate
      self.params['b1'] =self.params['b1'] -grads['b1']*learning_rate
      self.params['b2'] =self.params['b2'] -grads['b2']*learning_rate

                    #
      #########################################################################
      # pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # pass
    scores_firstLayer = X.dot(self.params['W1'])+self.params['b1']
    H = np.maximum(0,scores_firstLayer)
    scores_finalLayer =H.dot(self.params['W2']) +self.params['b2']
    y_pred = np.argmax(scores_finalLayer,axis=1)
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


