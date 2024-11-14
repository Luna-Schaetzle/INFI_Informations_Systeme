# Importieren der notwendigen Bibliotheken
import pandas as pd
import matplotlib.pyplot as plt

# 1. Daten einlesen und vorbereiten

# Pfad zur Excel-Datei (bitte anpassen)
dateipfad = 'Zeitreihe-Winter-2024011810.xlsx'

# Lesen der Excel-Datei
df = pd.read_excel(dateipfad, skiprows=[0, 2], header=0)

# Manuelles Benennen der Spalten
# Annahme: Die ersten drei Spalten sind 'Bez', 'Gemnr', 'Gemeinde'
# Die restlichen Spalten sind die Jahreszahlen von 2000 bis 2023
jahre = list(range(2000, 2024))
spaltennamen = ['Bez', 'Gemnr', 'Gemeinde'] + jahre
df.columns = spaltennamen

# 2. Berechnung der Gesamtzahl an Touristen pro Jahr

# Auswahl der Jahresspalten
jahres_spalten = jahre  # Die Jahreszahlen von 2000 bis 2023

# Berechnung der Gesamtzahl an Touristen pro Jahr (Summe über alle Gemeinden)
gesamt_pro_jahr = df[jahres_spalten].sum(axis=0)

# Ausgabe der Gesamtzahl pro Jahr
print("Gesamtzahl an Touristen pro Jahr:")
print(gesamt_pro_jahr)

# 3. Berechnung der Gesamtzahl über alle Jahre

gesamt_alle_jahre = gesamt_pro_jahr.sum()
print("\nGesamtzahl an Touristen über alle Jahre (2000-2023):", int(gesamt_alle_jahre))

# 4. Zusammenfassung nach Bezirken

# Gruppierung nach 'Bez' und Summierung der Jahresspalten
sum_bez = df.groupby('Bez')[jahres_spalten].sum()

# Ausgabe der Gesamtzahl pro Bezirk
print("\nGesamtzahl an Touristen pro Bezirk und Jahr:")
print(sum_bez)

# 5. Plotten der Zusammenfassung nach Bezirken

# Summe über alle Jahre für jeden Bezirk berechnen
sum_bez['Gesamt'] = sum_bez.sum(axis=1)

# Sortieren nach Gesamtzahl (optional)
sum_bez = sum_bez.sort_values('Gesamt', ascending=False)

# Plotten der Gesamtzahl pro Bezirk (über alle Jahre)
sum_bez['Gesamt'].plot.bar()
plt.title('Gesamtzahl der Touristen pro Bezirk (2000-2023)')
plt.xlabel('Bezirk')
plt.ylabel('Anzahl der Touristen')
plt.show()
