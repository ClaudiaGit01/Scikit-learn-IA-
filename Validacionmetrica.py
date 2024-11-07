from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# cargar el conjunto de datos Iris
iris = load_iris()
x_trail, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42) 

# Crear el clasificador de vecinos más cercanos
clf = KNeighborsClassifier(n_neighbors=3)
# Entrenar el clasificador
clf.fit(x_trail, y_train)
# Predecir las etiquetas para los dtos de prueba
y_pred = clf.predict(x_test)
# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print('Precisión:', accuracy)
print('Precisión promedio ponderada:', precision)
print('Recall:', recall)
print('F1 score:', f1)
