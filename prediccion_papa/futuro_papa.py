import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar archivo Excel
df = pd.read_excel("cardenales_papables.xlsx")

# Variables a usar
features = ['Edad', 'Región de Origen', 'Experiencia en la Curia',
            'Enfoque Pastoral', 'Idiomas Hablados', 'Alineación Teológica']

# Preprocesamiento
label_encoders = {}
for col in features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Inputs y simulación de etiquetas (1 si perfil se parece al Papa Francisco, 0 si no)
df['label'] = 0
df.loc[df.sort_values(by='Edad').head(5).index, 'label'] = 1  # Suponemos que los 5 más parecidos son positivos

# Normalización
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df['label'].values

# División del conjunto
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de red neuronal
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Probabilidad
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

# Predicciones
df['Probabilidad (%)'] = (model.predict(X).flatten() * 100).round(2)
df = df[['Nombre', 'Probabilidad (%)']].sort_values(by='Probabilidad (%)', ascending=False)

# Guardar resultados
df.to_excel("prediccion_papa_tensorflow.xlsx", index=False)
print(df.head(10))

#mostrar precision del modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
