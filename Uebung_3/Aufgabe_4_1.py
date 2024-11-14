# Importieren der notwendigen Bibliotheken
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# 1. Daten einlesen und vorbereiten

# Pfad zur Excel-Datei (bitte anpassen)
dateipfad = 'Zeitreihe-Winter-2024011810.xlsx'

# Lesen der Excel-Datei, überspringen der Zeilen 0 und 2, und setzen von Zeile 1 als Header
df = pd.read_excel(dateipfad, skiprows=[0, 2], header=0)

# Manuelles Benennen der Spalten
# Die ersten drei Spalten sind 'Bez', 'Gemnr', 'Gemeinde'
# Die restlichen Spalten entsprechen den Jahren von 2000 bis 2023
jahre = list(range(2000, 2024))
neue_spalten = ['Bez', 'Gemnr', 'Gemeinde'] + jahre
df.columns = neue_spalten

# Überprüfen der umbenannten Spalten
print("\nUmbenannte Spalten:")
print(df.columns)

# 2. Berechnung der standardisierten Ranges

# Identifizieren der Jahr-Spalten im DataFrame
jahres_spalten = [col for col in df.columns if isinstance(col, int) and 2000 <= col <= 2023]

# Berechnung des Range (max - min) pro Gemeinde
df['range'] = df[jahres_spalten].max(axis=1) - df[jahres_spalten].min(axis=1)

# Berechnung des Z-Score für den Range (Standardisierung)
df['range_z_score'] = (df['range'] - df['range'].mean()) / df['range'].std()

# Überprüfen der hinzugefügten Spalten
print("\nDataFrame nach Hinzufügen der standardisierten Range:")
print(df[['Bez', 'Gemnr', 'Gemeinde', 'range', 'range_z_score']].head())

# 3. Erstellung des Boxplots

# Methode a: Verwendung der Pandas-Boxplot-Methode
plt.figure(figsize=(14, 8))
df.boxplot(column='range_z_score', by='Bez', grid=False, vert=True, rot=90)
plt.title('Boxplot der standardisierten Ranges pro Bezirk')
plt.suptitle('')  # Entfernt den automatischen Supertitel
plt.xlabel('Bezirk')
plt.ylabel('Standardisierter Range (Z-Score)')
plt.tight_layout()
plt.show()

# Methode c: Verwendung von Seaborn für individuell gefärbte Bezirke
plt.figure(figsize=(14, 8))
sns.boxplot(x='Bez', y='range_z_score', data=df, palette='Set3')
plt.title('Boxplot der standardisierten Ranges pro Bezirk')
plt.xlabel('Bezirk')
plt.ylabel('Standardisierter Range (Z-Score)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
