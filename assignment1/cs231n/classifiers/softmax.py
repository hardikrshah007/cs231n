import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  i1 = X.shape[0]
  i2 = W.shape[1]
  # print(the shape of)
  for i in range(i1):
    d1 = X[i,:]
    sum_scores =0.0
    scores = d1.dot(W)
    correct_score = scores[y[i]]
    lossi = -np.log(np.exp(correct_score)/np.sum(np.exp(scores)))
    for j in range(i2):
      if j==y[i]:
        continue
      dW[:,j]+=np.exp(scores[j])*d1/np.sum(np.exp(scores))
    dW[:,y[i]]+=d1*(np.exp(scores[y[i]])-np.sum(np.exp(scores)))/np.sum(np.exp(scores))
    loss+=lossi
  loss = loss/i1+ reg*np.sum(W*W)
  dW/=i1
  dW+=2*reg*W


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                        #
  #############################################################################
  # pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  scoresnew = X.dot(W)
  scores_exp = np.exp(scoresnew)
  sum_scores_exp =np.sum(scores_exp,axis=1)
  correct_score =np.zeros(num_train)
  scores_exp_normalized = np.zeros_like(scores_exp)

  for i in range(num_train):
    correct_score[i]=scoresnew[i,y[i]]
    scores_exp_normalized[i,:] = scores_exp[i,:]/sum_scores_exp[i]

  loss_vector = -np.log(np.exp(correct_score)/sum_scores_exp)
  loss = np.sum(loss_vector)
  loss/=num_train +reg*np.sum(W*W)
  input_data_transpose = X.transpose()
  dW = input_data_transpose.dot((scores_exp_normalized))
  for i in range(num_train):
    dW[:,y[i]] += -X[i]
  dW/=num_train
  dW+=2*reg*W                                                           

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

