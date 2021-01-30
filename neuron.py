import numpy as np

def sigmoid(x, der=False):
	if der:
		return x*(1-x)
	return 1 / (1+np.exp(-x))

x=np.array([[1,0,1],[1,0,1],[0,1,0],[0,1,0]])
y=np.array([[0,0,1,1]]).T
np.random.seed(1)
SynapseZero= 2 * np.random.random((3, 1)) - 1
l1=[]
for iter in range(10000):
	l1=sigmoid(np.dot(x, SynapseZero))
	SynapseZero += np.dot(x.T, y - l1 * sigmoid(l1, True))
print("Output:")
print(l1)

new_one=np.array([1,0,1])
new=sigmoid(np.dot(new_one, SynapseZero))
print("new gen:")
print(new)
