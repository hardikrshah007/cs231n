#!/usr/bin/python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import os
from matplotlib.backends.backend_pdf import PdfPages
import math
from scipy.optimize import least_squares
import pyquaternion
from scipy.optimize import curve_fit


if __name__ == '__main__':

	# X_random= 0
	X= (6,2,11,4)
	
	sum_squareroot =np.sqrt(np.sum(np.square(X)))
	print(sum_squareroot)

	index = np.argsort(X)
	print(index)
	k =1
	for i in range(k):
		print(X[index[i]])
	# print(index);