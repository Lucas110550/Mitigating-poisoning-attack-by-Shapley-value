
import numpy as np
import scipy
from scipy.stats import mode
import pickle
import gzip


def get_value(trainX, trainy, valX, valy, K):
	N = trainX.shape[0]
	M = valX.shape[0]

	value = np.zeros(N)
	for i in range(M):
		X = valX[i]
		y = valy[i]

		s = np.zeros(N)
		diff = (trainX - X).reshape(N, -1)
		dist = np.einsum('ij, ij->i', diff, diff)
		idx = np.argsort(dist)
	#	print(idx)

		ans = trainy[idx]

	#	__debug(ans)
	#	print(idx.shape)
	#	print(ans.shape)
		s[idx[N - 1]] = float(ans[N - 1] == y) / N

		cur = N - 2
		for j in range(N - 1):
			s[idx[cur]] = s[idx[cur + 1]] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / K * (min(cur, K - 1) + 1) / (cur + 1)
			cur -= 1
			
		for j in range(N):
			value[j] += s[j]
	return value




	for i in range(N):
		value[j] /= M
