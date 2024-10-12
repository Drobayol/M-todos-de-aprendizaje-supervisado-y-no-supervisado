import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargar los datos
data = pd.read_csv('transporte_datos.csv')

# Visualizar los primeros registros
print("Datos originales:")
print(data.head())

# Convertir columnas categóricas en numéricas
data['Clima'] = data['Clima'].map({'soleado': 0, 'nublado': 1, 'lluvioso': 2})
data['Tráfico'] = data['Tráfico'].map({'bajo': 0, 'medio': 1, 'alto': 2})

# Separar características y etiquetas
X = data[['Num_Pasajeros', 'Clima', 'Tráfico']]
y = data['Puntualidad']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
