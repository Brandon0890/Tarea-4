import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, LeavePOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar los datos desde el archivo CSV
data = pd.read_csv('irisbin.csv')

# Separar características (X) y etiquetas (y)
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento, 20% para generalización)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un perceptrón multicapa (ajusta los parámetros según sea necesario)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = mlp.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%')

# Validación cruzada con leave-one-out
loo = LeaveOneOut()
loo_scores = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp.fit(X_train, y_train)
    y_pred_loo = mlp.predict(X_test)

    loo_accuracy = accuracy_score(y_test, y_pred_loo)
    loo_scores.append(loo_accuracy)

loo_average_accuracy = np.mean(loo_scores)
loo_std_accuracy = np.std(loo_scores)

print(f'Leave-one-out - Precisión promedio: {loo_average_accuracy * 100:.2f}%, Desviación estándar: {loo_std_accuracy * 100:.2f}%')

# Validación cruzada con leave-p-out (puedes ajustar el número de p según sea necesario)
lpout = LeavePOut(p=5)
lpout_scores = []

for train_index, test_index in lpout.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp.fit(X_train, y_train)
    y_pred_lpout = mlp.predict(X_test)

    lpout_accuracy = accuracy_score(y_test, y_pred_lpout)
    lpout_scores.append(lpout_accuracy)

lpout_average_accuracy = np.mean(lpout_scores)
lpout_std_accuracy = np.std(lpout_scores)

print(f'Leave-p-out - Precisión promedio: {lpout_average_accuracy * 100:.2f}%, Desviación estándar: {lpout_std_accuracy * 100:.2f}%')
