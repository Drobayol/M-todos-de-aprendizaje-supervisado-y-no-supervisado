import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar los datos
data = pd.read_csv('transporte_datos.csv')

# Convertir columnas categóricas en numéricas
data['Clima'] = data['Clima'].map({'soleado': 0, 'nublado': 1, 'lluvioso': 2})
data['Tráfico'] = data['Tráfico'].map({'bajo': 0, 'medio': 1, 'alto': 2})

# Seleccionar características para el clustering
X = data[['Num_Pasajeros', 'Clima', 'Tráfico']]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Obtener etiquetas de los clusters
data['Cluster'] = kmeans.labels_

# Visualización de los clusters
plt.scatter(data['Num_Pasajeros'], data['Clima'], c=data['Cluster'], cmap='viridis')
plt.title('Clusters de pasajeros según Clima y Número de Pasajeros')
plt.xlabel('Número de Pasajeros')
plt.ylabel('Clima')
plt.colorbar()
plt.show()
# Desde la terminal se ejecuta el codigo bajo el comando: python modelo_transporte_no_supervisado.py
