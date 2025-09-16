import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Cargar el archivo original con los datos de los cardenales
df = pd.read_excel("cardenales_papables.xlsx")

# Definir las características clave
features = ['Edad', 'Región de Origen', 'Experiencia en la Curia',
            'Enfoque Pastoral', 'Idiomas Hablados', 'Alineación Teológica']

# Codificar variables categóricas
label_encoders = {}
for col in features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Construir el vector del perfil ideal del Papa Francisco
perfil_ideal = {
    'Edad': 76,
    'Región de Origen': 'América (Argentina)',
    'Experiencia en la Curia': 'Media',
    'Enfoque Pastoral': 'Progresista',
    'Idiomas Hablados': 'Español',
    'Alineación Teológica': 'Progresista'
}

# Convertir el perfil ideal a valores codificados
ideal_vector = []
for feature in features:
    if feature in label_encoders:
        encoded_value = label_encoders[feature].transform([perfil_ideal[feature]])[0]
    else:
        encoded_value = perfil_ideal[feature]
    ideal_vector.append(encoded_value)

# Crear matriz de entrada y normalizar
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ideal_scaled = scaler.transform([ideal_vector])

# Calcular distancia euclidiana y similitud
distancias = euclidean_distances(X_scaled, ideal_scaled).flatten()
df['Distancia al Perfil Ideal'] = distancias
df['Similitud (%)'] = (1 / (1 + distancias)) * 100

# Guardar solo nombre y similitud
df_resultado = df[['Nombre', 'Similitud (%)']].sort_values(by='Similitud (%)', ascending=False)
df_resultado.to_excel("similitud_papa_francisco.xlsx", index=False)

print("✅ Archivo generado: similitud_papa_francisco.xlsx")
