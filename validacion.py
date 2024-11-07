from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# cargar el conjunto de datos Iris
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

print('numero de muestras en el conjunto de entrenamiento:', len(x_train))
print('numero de muestras en el conjunto de prueba:', len(x_test))

