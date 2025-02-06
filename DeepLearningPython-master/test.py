"""
    Testing code for different neural network configurations.
    Adapted for Python 3.5.2

    Usage in shell:
        python3.5 test.py

    Network (network.py and network2.py) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Michał Dobrzański, 2016
        dobrzanski.michal.daniel@gmail.com
"""

# ----------------------
# - read the input data:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
# Selección de la Red Neuronal
print("Selecciona la red neuronal que quieras ejecutar:")
print("1. Básica con 30 neuronas ocultas y MSE")
print("2. Básica con 100 neuronas ocultas y MSE")
print("3. 30 neuronas ocultas y Cross Entropy con regularización L2 lambda=5 e inicialización de pesos normal")
print("4. 30 neuronas ocultas y Cross Entropy con regularización L2 lambda=5 e inicialización de pesos distorsionada")
print("5. 100 neuronas ocultas y Cross Entropy con regularización L2 lambda=5 e inicialización de pesos normal")
print("6. 100 neuronas ocultas y Cross Entropy con regularización L2 lambda=5 e inicialización de pesos distorsionada")
print("7. Red Neuronal Profunda y Cross Entropy con regularización L2 lambda=5 e inicialización de pesos distorsionada")
n = int(input(""))

if n==1:
    # network.py - Red nejuronal básica con 30 neuronas en la capa oculta y MSE:
    import network
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    # Alcanza un 95.14% de precisión en el mejor de tres intentos
elif n==2:
    # network.py - Red nejuronal básica con 100 neuronas en la capa oculta y MSE:
    import network
    net = network.Network([784, 100, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    # Alcanza un 87,37%

elif n==3:
    # network2.py - 30 neuronas ocultas con regularización L2 lambda=5 e inicialización de pesos normal
    import network2
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, early_stopping_n=10)
    # Alcanza un 96.38%
elif n==4:
    # network2.py - 30 neuronas ocultas con regularización L2 lambda=5 e inicialización de pesos distorsionada
    import network2
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, early_stopping_n=10)
    # Alcanza un 96.54%
elif n==5:
    # network2.py - 100 neuronas ocultas con regularización L2 lambda=5 e inicialización de pesos normal
    import network2
    net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 60, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, early_stopping_n=10)
    # Alcanza un 98.13% (97.81% si epoch=30 y eta=0.5) 
elif n==6:
    # network2.py - 100 neuronas ocultas con regularización L2 lambda=5 e inicialización de pesos distorsionada
    import network2
    net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
    net.SGD(training_data, 60, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, early_stopping_n=10)
    # Alcanza un 98.21%
elif n==7:
    # network2.py - Red Neuronal profunda con regularizacion L2 lambda=5 e inicialización de pesos distorsionada
    import network2
    net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        early_stopping_n=20)
    # Alcanza un %