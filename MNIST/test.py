import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

print("Hola")
# Selección de la Red Neuronal a ejecutar
print("Selecciona la red neuronal que quieras ejecutar:")
print("1. 100 neuronas ocultas: Optimizador SGD, Cross Entropy Cost, sigmoide, eta=0.1, minibach_size=10, epoch=60, regularización L2 (lambda=0.0001) e inicialización de pesos distorsionada")
print("2. 100 neuronas ocultas: Optimizador SGD, MSE, sigmoide, eta=0.5, minibach_size=10, epoch=60, regularización L2 (lambda=0.0001) e inicialización de pesos distorsionada")
print("3. 100 neuronas ocultas: Optimizador SGD, Cross Entropy Cost, ReLU, eta=0.1, minibach_size=10, epoch=60, regularización L2 (lambda=0.0001) e inicialización de pesos distorsionada")
print("4. 200 neuronas ocultas: Optimizador Adam (parámetros por defecto), Cross Entropy Cost, ReLU, eta=0.002, minibach_size=64, epoch=60, regularización weight decay (lambda=0.0001) e inicialización de pesos distorsionada")
n = int(input(""))

if n==1:
    # network.py - 100 neuronas ocultas: Optimizador SGD, Cross Entropy Cost, sigmoide, eta=0.1, minibach_size=10, epoch=60, regularización L2 (lambda=0.0001) e inicialización de pesos distorsionada
    import network
    net = network.Network([784, 100, 10], cost=network.CrossEntropyCost)
    net.SGD(training_data, 60, 10, 0.1, lmbda = 0.0001,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, early_stopping_n=10)
    # Alcanza un 98.02%

elif n==2:
    # network.py - 100 neuronas ocultas: Optimizador SGD, MSE, sigmoide, eta=0.5, minibach_size=10, epoch=60, regularización L2 (lambda=0.0001) e inicialización de pesos distorsionada
    import network
    net = network.Network([784, 100, 10], cost=network.QuadraticCost)
    net.SGD(training_data, 60, 10, 0.5, lmbda = 0.0001,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, early_stopping_n=10)
    # Alcanza un 97.08%

elif n==3:
    # network.py - 100 neuronas ocultas: Optimizador SGD, Cross Entropy Cost, ReLU, eta=0.1, minibach_size=10, epoch=60, regularización L2 (lambda=0.0001) e inicialización de pesos distorsionada
    import network
    net = network.Network([784, 100, 10], cost=network.CrossEntropyCost)
    net.SGD(training_data, 60, 10, 0.1, lmbda = 0.0001,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, early_stopping_n=10)
    # Alcanza un 98.14%
    
elif n==4:
    # network_Adam.py - 200 neuronas ocultas: Optimizador Adam (parámetros por defecto), Cross Entropy Cost, sigmoide, eta=0.002, minibach_size=64, epoch=60, regularización weight decay (lambda=0.0001) e inicialización de pesos distorsionada
    import network_Adam
    net = network_Adam.Network([784, 200, 10], cost=network_Adam.CrossEntropyCost, activation=network_Adam.ReLUActivation)
    net.SGD(training_data, 100, 64, 0.002, lmbda = 0.0001,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, early_stopping_n=20)
    # Alcanza un 98,26%