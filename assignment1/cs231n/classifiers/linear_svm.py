import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    loss_count =0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j]+= X[i]
        loss_count+=1
    dW[:,y[i]]+=-1*loss_count*X[i]	 
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW/= num_train
  

  # Add regularization to the loss.
   #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  loss += reg * np.sum(W * W)
  dW+=reg*2*W

  return loss, dW



def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  correct_class_score = np.zeros(y.shape)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  scores=X.dot(W)
  for i in range(num_train):
  	correct_class_score[i] = scores[i,y[i]]
  margin = scores - correct_class_score[:,None] + 1
  for i in range(num_train):
  	margin[i,y[i]] = 0
  pos_num = margin>0
  pos_num_0 = np.sum(pos_num,axis=1)
  pos_num_1=np.sum(pos_num,axis=0)
  # print("the shape of only_pos is: ",pos_num_0.shape)
  
  pos_num_transpose= pos_num.transpose()
  # for i in range(num_classes):
  # 	pos_single = pos_num_transpose[i,:]
  # 	dW[:,i] = pos_single.dot(X)
  dW_transpose= pos_num_transpose.dot(X)
  dW= dW_transpose.transpose()
  for i in range(num_train):
  	dW[:,y[i]] +=-1*pos_num_0[i]*X[i]
  loss = np.sum(margin[margin>0])
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW/= num_train
  dW+=reg*2*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
