import numpy as np

parser = argparse.ArgumentParser(description = None)
parser.add_argument('--num', type=int, required = True)
args = parser.parse_args()

poison_num = args.num

f = open("resultnew.txt", "r")
m = poison_num
rubbish = np.zeros((600, m))
mark = False
count = 0
xx = []
truth = False

def isdight(x):
	if (len(x)) == 0:
		return False
	if (x[0] < '0' or x[0] > '9'):
		return False
	return True

for s in f:
	p = s.split(" ")
	if (len(p) >= 3):
		if (p[0] == '***' and p[1] == 'Iter:' and p[2] == '0\n'):
			mark = False
	if (mark == True):
		for i in range(len(p)):
			if (isdight(p[i])):
				rubbish[count - 1][rm] = int((p[i].split("]"))[0])
				rm += 1
	elif (p[0] == 'Test' and p[1] == 'idx:'):
		mark = True
		count += 1
		rm = 0
	if (p[0] == 'Good' and p[1] == 'data,'):
		truth = True
	if (p[0] == 'Current' and p[1] == 'Success'):
		if (truth == True):
			xx.append(count - 1)
		truth = False

import pickle
print(rubbish)
print(len(xx))
with open("poison.pkl", "wb") as tf:
	pickle.dump(rubbish[xx], tf)


