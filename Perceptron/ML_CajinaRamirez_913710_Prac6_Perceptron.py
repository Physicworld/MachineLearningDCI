'''

---------------------------------------------------------

Domingo Cajina 24/10/2022

---------------------------------------------------------

'''

import numpy as np
import matplotlib.pyplot as plt


sigmoid = lambda x: 1/(1 + np.e**-x)

tanh = 	lambda x: np.tanh(x)

relu = 	lambda x: np.maximum(0, x)

def random_points(n = 100):
	x = np.random.uniform(0.0, 1.0, n)
	y = np.random.uniform(0.0, 1.0, n)

	return np.array([x, y]).T

class Perceptron:

	def __init__(self, n_inputs, act_f):
		'''
		Inicializamos pesos, el bias y la funcion de activacion,
		'''
		self.weights = np.random.rand(n_inputs,1)
		self.bias = np.random.rand()
		self.act_f = act_f
		self.n_inputs = n_inputs

	def predict(self, x):
		'''
		Metodo predict, realiza el producto punto entre
		las entradas y los pesos, suma el bias y evalua en
		la funcion de activacion.
		'''
		return self.act_f(x @ self.weights + self.bias)

	def fit(self, x, y, epochs = 100, learning_rate = 0.5):
		'''
		Metodo fit, se encarga de entrenar al perceptron,
		calculando el error en cada iteracion y ajustando
		los pesos y el bias.

		Podemos entrenar hasta que el error sea 0,
		pero no es recomendable por que tenemos mucho
		riesgo de overfitting.
		'''
		for i in range(epochs):
			for j in range(len(x)):
				output = self.predict(x[j])
				error = (y[j] - output) * learning_rate
				self.weights = self.weights + (error * x[j][1])
				self.bias = self.bias + error




def main():
	points = random_points(10000)
	plt.scatter(points[:,0], points[:,1], s = 10)
	# plt.show()

	'''
	COMPUERTA OR
	'''

	x = np.array([
				[0,0],
				[0,1],
				[1,0],
				[1,1]
	])
	
	y = np.array([
		[0],
		[1],
		[1],
		[1]
	])

	p_or = Perceptron(2, sigmoid)

	yp = p_or.predict(points)
	plt.scatter(points[:,0], points[:,1], s = 10, c=yp, cmap='GnBu')
	plt.title("PERCEPTRON SIN ENTRENAR")
	# plt.show()
	plt.savefig('Perceptron sin entrenar')

	p_or.fit(x = x, y = y, epochs=1000, learning_rate=0.5)

	yp = p_or.predict(points)
	plt.scatter(points[:,0], points[:,1], s = 10, c=yp, cmap='GnBu')
	plt.colorbar(label="Probability") 
	plt.title("OR PERCEPTRON")
	# plt.show()
	plt.savefig('Perceptron entrenado OR')

	'''
	COMPUERTA AND
	'''

	x = np.array([
				[0,0],
				[0,1],
				[1,0],
				[1,1]
	])
	
	y = np.array([
		[0],
		[0],
		[0],
		[1]
	])

	p_and = Perceptron(2, sigmoid)
	p_and.fit(x = x, y = y, epochs=1000, learning_rate=0.4)
	yp = p_and.predict(points)
	plt.scatter(points[:,0], points[:,1], s = 10, c=yp, cmap='GnBu')
	# plt.colorbar(label="Probability") 
	plt.title("AND PERCEPTRON")
	# plt.show()
	plt.savefig('Perceptron entrenado AND')


	


if __name__ == '__main__':
	main()
	
