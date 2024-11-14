# Importieren der notwendigen Bibliotheken
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Daten einlesen und vorbereiten

# 1.1 Pfade zu den Excel-Dateien (bitte anpassen)
dateipfad = 'Zeitreihe-Winter-2024011810.xlsx'
einwohner_dateipfad = 'Einwohnerzahlen.xls'  # Einwohnerdaten

# 1.2 Einlesen der Touristikdaten
df_tourismus = pd.read_excel(tourismus_dateipfad, skiprows=[0, 2], header=0)

# Umbenennen der Spalten
jahre = list(range(2000, 2024))
neue_spalten_tourismus = ['Bez', 'Gemnr', 'Gemeinde'] + jahre
df_tourismus.columns = neue_spalten_tourismus

# 1.3 Einlesen der Einwohnerdaten
df_einwohner = pd.read_excel(einwohner_dateipfad)

# Überprüfen der Einwohnerdaten
print("\nEinwohnerdaten - Erste fünf Zeilen:")
print(df_einwohner.head())

# 1.4 Zusammenführen der beiden Datensätze anhand von 'Gemnr'
df_both = pd.merge(df_tourismus, df_einwohner, how='inner', on='Gemnr')

# 1.5 Bereinigen der zusammengeführten Daten
# Entfernen doppelter Spalten (falls vorhanden)
df_both = df_both.loc[:, ~df_both.columns.duplicated()]

# 1.6 Überprüfen der zusammengeführten Daten
print("\nZusammengeführte Daten - Erste fünf Zeilen:")
print(df_both.head())

# 2. Berechnungen durchführen

# 2.1 a. Standardisierung der Anzahl Nächtigungen im Jahr 2018 mit der Bevölkerung pro Gemeinde

# Überprüfen der verfügbaren Spalten
print("\nVerfügbare Spalten in den zusammengeführten Daten:")
print(df_both.columns)

# Beispiel: Angenommen, die Einwohnerzahl ist in der Spalte 'Einwohner' gespeichert
# Berechnung der Touristen pro Einwohner im Jahr 2018
df_both['Touristen_pro_Einwohner_2018'] = df_both[2018] / df_both['Einwohner']

# Überprüfen der Berechnung
print("\nBerechnung der Touristen pro Einwohner im Jahr 2018 - Erste fünf Zeilen:")
print(df_both[['Bez', 'Gemnr', 'Gemeinde', 2018, 'Einwohner', 'Touristen_pro_Einwohner_2018']].head())

# 2.2 b. Darstellung dieser Zahl als Boxplot, gruppiert nach Bezirk

plt.figure(figsize=(14, 8))
sns.boxplot(x='Bez', y='Touristen_pro_Einwohner_2018', data=df_both, palette='Set3')
plt.title('Touristen pro Einwohner in 2018 nach Bezirk', fontsize=16)
plt.xlabel('Bezirk', fontsize=14)
plt.ylabel('Touristen pro Einwohner', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2.3 c. Die 10 Gemeinden mit dem größten und kleinsten Verhältnis

# Sortieren nach 'Touristen_pro_Einwohner_2018' in absteigender Reihenfolge
df_sorted = df_both.sort_values('Touristen_pro_Einwohner_2018', ascending=False)

# Die 10 Gemeinden mit dem größten Verhältnis
df_high = df_sorted.head(10)
print("\nTop 10 Gemeinden mit dem größten Touristen-Einwohner-Verhältnis in 2018:")
print(df_high[['Bez', 'Gemnr', 'Gemeinde', 'Touristen_pro_Einwohner_2018']])

# Die 10 Gemeinden mit dem kleinsten Verhältnis
df_low = df_sorted.tail(10)
print("\nTop 10 Gemeinden mit dem kleinsten Touristen-Einwohner-Verhältnis in 2018:")
print(df_low[['Bez', 'Gemnr', 'Gemeinde', 'Touristen_pro_Einwohner_2018']])

# 2.4 d. Verhältnis in der Heimatgemeinde

# Annahme: 'Heimatgemeinde' ist eine spezifische Gemeinde, z.B. 'Innsbruck'
# Bitte ersetzen Sie 'Innsbruck' mit dem tatsächlichen Namen Ihrer Heimatgemeinde
heimatgemeinde_name = 'Innsbruck'  # Ersetzen Sie dies mit dem tatsächlichen Namen

# Filtern der Heimatgemeinde
df_heimat = df_both[df_both['Gemeinde'].str.strip().str.lower() == heimatgemeinde_name.lower()]

# Überprüfen, ob die Heimatgemeinde gefunden wurde
if df_heimat.empty:
    print(f"\nFehler: Die Heimatgemeinde '{heimatgemeinde_name}' wurde im Datensatz nicht gefunden.")
else:
    # Anzeigen des Verhältnisses
    ratio_heimat = df_heimat['Touristen_pro_Einwohner_2018'].values[0]
    print(f"\nTouristen pro Einwohner in der Heimatgemeinde '{heimatgemeinde_name}' im Jahr 2018: {ratio_heimat:.4f}")
