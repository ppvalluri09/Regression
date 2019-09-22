import numpy as np
from matplotlib import pyplot as plt

def prepareData(X, type):
	if type == 'f' or type == 'features':
	    padding = np.ones((len(X), 1)).T[0]
	    X_ = np.array(X)
	    X_train = np.array([padding, X_])
	    return X_train.T
	elif type == 'l' or type == 'labels':
		y_ = np.array([X]).T
		return y_


def sigmoid(X):
	return 1 / (1 + np.exp(-X))

def sort(a, b):
	for i in range(len(a)):
		for j in range(len(b)):
			if i != j and a[i] < a[j]:
				a[i], a[j] = a[j], a[i]
				b[i], b[j] = b[j], b[i]

	return a, b

def best_fit(X, y, y_pred):
	plt.scatter(X, y, s=10, c='b')
	plt.xlabel('Features')
	plt.ylabel('Labels')
	plt.title('Logit')
	X, y_pred = sort(X, y_pred)
	plt.plot(X, y_pred, c='r')
	# plt.show()

class Model:

	def __init__(self, dim=1):
		self.R2 = 0
		self.w = np.random.randn(dim, 1)
		self.prediction = 0
		self.X = 0
		self.y = 0
		self.mean = []
		self.var = []
		self.cost = []
		self.cycles = 1

	def scale(self, data):
		data = data.T
		if len(self.mean) == 0:
			for i in range(1, len(data)):
				self.mean.append(np.mean(data[i]))
				self.var.append(np.std(data[i]) ** 2)
		for i in range(len(self.mean)):
			data[i + 1] = (data[i + 1] - self.mean[i]) / self.var[i]

		return data.T

	def fit(self, X):
		return sigmoid(np.dot(X, self.w))

	def transform(self, X, y, cycles=50000):

		self.X = X
		self.y = y
		self.cost = []
		self.cycles = cycles
		self.w = np.random.randn(len(X[0]), len(y[0]))	
		learning_rate = 0.07
		m = len(X)

		for iterations in range(cycles):
			y_pred = self.fit(X)
			cost = self.calculate_cost(y_pred)
			self.cost.append(cost)
			error = (1/m) * np.dot(X.T, (y_pred - y))
			self.w = self.w - learning_rate * error

	def calculate_cost(self, y_pred):
		m = len(self.y)
		return -1.0 * (1 / m) * np.sum((self.y * (np.log(y_pred))) + ((1 - self.y) * (np.log(1 - y_pred))))

	def plot_cost(self):
                x = [i for i in range(self.cycles)]
                plt.xlabel('Iterations')
                plt.ylabel('Cost')
                plt.title('Cost vs Iterations')
                plt.plot(x, self.cost, c='g')
                plt.show()


X = [24, 52, 10, 5, 3, 17, 18, 19, 27, 45, 36, 71, 1, 8, 14, 6]
X_train = prepareData(X, 'f')
y = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
y_train = prepareData(y, 'l')

logit = Model()
X_train = logit.scale(X_train)
logit.transform(X_train, y_train)
y_pred = logit.fit(X_train)

best_fit(X, y, y_pred.T[0])

X_test = [34, 16, 45, 12]
X_test_ = prepareData(X_test, 'f')
logit.scale(X_test_)
y_test = logit.fit(X_test_).T
plt.scatter(X_test, y_test[0], c='g')

print(y_test)
print(logit.w)
plt.show()

logit.plot_cost()
