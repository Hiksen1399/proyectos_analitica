import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Leer el archivo Excel con los resultados de similitud
df = pd.read_excel("similitud_papa_francisco.xlsx")

# Seleccionar los 10 cardenales más similares
top_10 = df.head(10)

# Agregar manualmente el perfil ideal del Papa Francisco
papa_francisco_entry = {
    'Nombre': 'Papa Francisco (Ideal)',
    'Similitud (%)': 100.0
}

# Insertar al inicio del DataFrame
top_10_with_papa = pd.concat([pd.DataFrame([papa_francisco_entry]), top_10], ignore_index=True)

# Crear gráfico de barras
plt.figure(figsize=(12, 6))
bars = plt.bar(top_10_with_papa['Nombre'], top_10_with_papa['Similitud (%)'], color=['gold'] + ['skyblue'] * 10)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Similitud al Perfil del Papa Francisco (%)')
plt.title('Top 10 Cardenales Más Similares al Papa Francisco (posible sucesor)')
plt.tight_layout()

# Resaltar la barra del Papa Francisco
bars[0].set_color('green')

# Mostrar la gráfica
plt.show()

