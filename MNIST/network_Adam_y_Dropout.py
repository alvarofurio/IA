# %% [markdown]
# # Red Neuronal: Clasificación de dígitos escritos a mano
# Se propone una red neuronal básica que implementa el SGD (posteriormente el optimizador ADAM), la función de coste cross entropy y MSE (posteriormente huber loss), la regularizacion L1 y L2 (ligeramente modificada), y una mejor inicialización de los pesos

# %% [markdown]
# #### Importamos las librerias

# %%
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np

# %% [markdown]
# #### Funciones de coste y activación
# Definimos las funciones de coste entre vectores (capa de salida y target) y el error imputado por la ultima capa (delta). 
# $$C_{x_i} := \frac{1}{2}{\| a^{(L)} - \hat{y}\|}_{2}^2 \hspace{1.5cm} {\large \frac{\partial C_{x_i}}{\partial a^{(L)}}} = a^{(L)} - \hat{y} $$

# %% [markdown]
# 
# También definimos la función de activación sigmoide
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

# %%
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y)*sigmoid_prime(z)
    # Cambiar el sigmoid_prime bajo

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        # np.nan_to_num se utiliza para convertir el 0·ln(0) en 0
        # return np.sum(np.nan_to_num(-y * np.log(a)))
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    @staticmethod
    def delta(z, a, y):
        return (a-y)
    

def sigmoid(z):
    # Esta implementado de esta forma para evitar overflows en las primeras iteraciones
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return np.maximum(0,z)

def relu_prime(z):
    return np.where(z <= 0, 0, 1)

# %% [markdown]
# #### Función auxiliar

# %%
def vectorized_result(j):
    """
    Vectoriza un índice j en un vector one-hot de 10 dimensiones con un 1 en la posición j
    y 0 en el resto
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# %% [markdown]
# ## Clase Principal

# %%
class Network(object):

    def __init__(self, sizes, cost=QuadraticCost, dropout_p=0.5):
        """
        La lista 'sizes' contiene el número de neuronas en cada capa
        Los pesos y biases están inicializados aleatoriamente
        dropout_p puede ser un solo valor o una lista de valores para cada capa
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        if isinstance(dropout_p, list):
            assert len(dropout_p) == self.num_layers - 1, "Length of dropout_p list must be equal to number of layers minus 1"
        self.dropout_p = dropout_p

        # Inicializamos m y v para Adam
        self.m_b = [np.zeros(b.shape) for b in self.biases]
        self.v_b = [np.zeros(b.shape) for b in self.biases]
        self.m_w = [np.zeros(w.shape) for w in self.weights]
        self.v_w = [np.zeros(w.shape) for w in self.weights]
        self.t = 0

        self.masks = []

    def default_weight_initializer(self):
        """
        Inicializamos los pesos con una gaussiana de media 0 y desviacion tipica 1/sqrt(nºneuronas cpa anterior)
                weights es un tensor en el que la matriz i-ésimo es la matriz de pesos de la capa i
        Inicializamos los biases con una gaussiana de media 0 y desviación típica 1
                biases es una matriz en el que la columna i-ésima es el vector de bias de la capa i
        Obviamente la capa de entrada no tiene pesos ni sesgos asociados

        La distorsión de la campana de Gauss suele tener mejores resultados que la simple
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def simple_weight_initializer(self):
        """
        Inicializamos los pesos con una gaussiana de media 0 y desviacion tipica 1/sqrt(nºneuronas cpa anterior)
        Inicializamos los biases con una gaussiana de media 0 y desviación típica 1
        Obviamente la capa de entrada no tiene pesos ni sesgos asociados
        """

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Devuelve el output de la red siendo 'a' el input sin aplicar dropout"""
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            p = self.dropout_p[i] if isinstance(self.dropout_p, list) else self.dropout_p
            a = p*sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):
        """
        - 'training_data' es una lista de tuplas (x,y) donde x es un vector de entrada (pixeles imagen) e y el vector target esperado (del 0 al 9)
        - 'lambda' es el parámetro de regularización, por defecto a 0, es decir, sin regularización
        - El resto de argumentos son para monitorizar el coste y la precisión al final de cada iteración
        """

        # early stopping functionality:
        best_accuracy = 1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # Bucle principal, ciclos
        for j in range(epochs):
            # Creamos los mini-batches aleatoriamente
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            self.masks = []
            for i in range(self.num_layers):
                if i < self.num_layers - 2:  # No aplicamos dropout en la última capa
                    p = self.dropout_p[i] if isinstance(self.dropout_p, list) else self.dropout_p
                    mask = np.random.binomial(1, p, size=(self.sizes[i],1))
                    self.masks.append(mask)

            # Actualizamos los pesos por cada mini-batch
            for mini_batch in mini_batches:
                self.t += 1
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)

            # Monitorización
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Actualiza los parámetros de la red aplicando el algoritmo de backpropagation a un solo mini-batch,
        el cual consiste en una lista de tuplas (x,y) donde x es un vector de entrada (pixeles imagen) e y el vector target esperado (del 0 al 9)
        n es el número total de datos de entrenamiento.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Calculamos la media de los gradientes
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb/len(mini_batch) for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw/len(mini_batch) for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Actualizamos los pesos y biases usando Adam con regularización L2
        for i, (w, b, nw, nb) in enumerate(zip(self.weights, self.biases, nabla_w, nabla_b)):
            # Actualizamos m y v para los biases
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * nb
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (nb ** 2)
            # Calculamos las correcciones bias-corrected
            m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)
            # Actualizamos biases
            self.biases[i] = b - eta * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            # Actualizamos m y v para los pesos
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * nw
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (nw ** 2)
            # Calculamos las correcciones bias-corrected
            m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
            # Actualizamos pesos con regularización L2
            self.weights[i] = w - (eta * m_w_hat / (np.sqrt(v_w_hat) + epsilon) + lmbda*w)

    def backprop(self, x, y):
        """
        Devuelve la tupla (nabla_b, nabla_w) que representa el gradiente de la función de coste
        C_x para una sola muestra, en el mismo formato que los pesos y biases.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # PROPAGACIÓN DIRECTA (para obtener todos los z's y a's)
        activations = [x]   # lista de todas las activaciones, i.e., el elemento i-ésimo será a^(i)
        zs = []             # lista de todas las z's,i.e., el elemento i-ésimo será z^(i)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[-1])+b
            zs.append(z)
            activations.append(sigmoid(z))

        for i in range(len(activations)):
            if i < self.num_layers - 2:  # No aplicamos dropout en la última capa
                activations[i] *= self.masks[i]
                


        # RETROPROPAGACIÓN
        # Caso base l=L
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Caso inductivo l<L
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        - 'data' es una tupla (x,y) donde x es un vector de entrada (pixeles imagen) e y el vector target esperado (del 0 al 9)
        - 'convert' a False implica que se realiza sobre los casos de test y a True sobre los casos de entrenamiento
        La función devuelve el número de datos que la red predice correctamente. Se considera el output
        de la red se considera la neurona de la última capa cuya activación es mayor
        """
        if convert:
            # Datos de entrenamiento: se asume que y es un vector codificado one-hot
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            # Datos de validación: se asume que y es directamente índice
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        - 'data' es una tupla (x,y) donde x es un vector de entrada (pixeles imagen) e y el vector target esperado (del 0 al 9)
        - 'convert' a True implica que se realiza sobre los casos de test y a False sobre los casos de entrenamiento
        La función devuelve el número de datos que la red predice correctamente. Se considera el output
        de la red se considera la neurona de la última capa cuya activación es mayor
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

# %% [markdown]
# ### Carga de la Red

# %%
def load(filename):
    """
    Load a neural network from the file ``filename``. Devuelve una instancia de la clase Network
    """

    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
