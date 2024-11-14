# Importieren der notwendigen Bibliotheken
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# 1. Importieren und Aufbereiten des Datensatzes

# 1.1 Beschreibung des Datensatzes
# Der Datensatz enthält die Anzahl der Nächtigungen in den Wintersaisonen von 2000 bis 2023,
# aufgegliedert nach Gemeinden.

# 1.2 Aufbereiten der Daten

# Pfad zur Excel-Datei
dateipfad = 'Zeitreihe-Winter-2024011810.xlsx'

# Lesen der Excel-Datei ohne Zeilen zu überspringen und mit der ersten Zeile als Header
df = pd.read_excel(dateipfad, header=0)

# Überprüfen, ob die Daten korrekt eingelesen wurden
print("Erste fünf Zeilen des eingelesenen Datensatzes:")
print(df.head())

# Entfernen von Zeilen, die vollständig aus NaN-Werten bestehen
df = df.dropna(how='all')

# Überprüfen der verbleibenden Zeilen
print("\nDatensatz nach dem Entfernen von vollständig leeren Zeilen:")
print(df.head())

# 1.3 Import und Kontrolle

# Überprüfen der Spaltennamen und deren Typen
print("\nSpaltennamen und deren Typen vor der Umbenennung:")
for col in df.columns:
    print(f"{col} (Typ: {type(col)})")

# Umbenennen der Jahr-Spalten: Präfix 'x' hinzufügen und sicherstellen, dass sie als Strings behandelt werden
# Identifizieren der Jahr-Spalten (angenommen, sie sind numerisch und liegen zwischen 2000 und 2023)
jahres_spalten = [col for col in df.columns if isinstance(col, (int, float)) and 2000 <= col <= 2023]

# Erstellen eines Dictionaries für die Umbenennung (float zu 'x' + str ohne Dezimalstellen)
umbenennung_dict = {jahr: f'x{int(jahr)}' for jahr in jahres_spalten}

# Umbenennen der Spalten
df.rename(columns=umbenennung_dict, inplace=True)

# Überprüfen der umbenannten Spalten
print("\nUmbenannte Spalten:")
print(df.columns)

# Kontrolle der Daten mit describe()
beschreibung = df.describe()

# Verwenden von tabulate für eine bessere Ausgabe
print("\nBeschreibende Statistik des Datensatzes:")
print(tabulate(beschreibung, headers='keys', tablefmt='psql'))

# 2. Erste Auswertung

# 2.1 Wachstum darstellen

# Filtern der Zeile für Innsbruck
# Annahme: Die Gemeinde "Innsbruck" ist eindeutig identifiziert
# Überprüfen, ob die Spalte 'Gemeinde' vorhanden ist
if 'Gemeinde' not in df.columns:
    print("\nFehler: Die Spalte 'Gemeinde' ist im Datensatz nicht vorhanden.")
else:
    innsbruck_df = df[df['Gemeinde'].str.strip() == 'Innsbruck']

    # Überprüfen, ob Innsbruck gefunden wurde
    if innsbruck_df.empty:
        print("\nFehler: Die Gemeinde 'Innsbruck' wurde im Datensatz nicht gefunden.")
    else:
        # Extrahieren der Jahreszahlen und der entsprechenden Nächtigungen
        jahre = [f'x{jahr}' for jahr in range(2000, 2024)]
        
        # Überprüfen, ob alle benötigten Jahres-Spalten vorhanden sind
        fehlende_spalten = [jahr for jahr in jahre if jahr not in innsbruck_df.columns]
        if fehlende_spalten:
            print(f"\nFehler: Die folgenden Jahres-Spalten fehlen im Datensatz: {fehlende_spalten}")
        else:
            naechitungen = innsbruck_df[jahre].values.flatten()

            # Erstellen einer Liste der Jahre für die x-Achse (ohne das 'x' Präfix)
            jahre_numeric = list(range(2000, 2024))

            # Plotten des zeitlichen Verlaufs als Punktdiagramm mit 'i' als Marker
            plt.figure(figsize=(12, 6))
            plt.plot(jahre_numeric, naechitungen, marker='i', linestyle='-', color='blue')
            plt.title('Zeitlicher Verlauf der Nächtigungen in Innsbruck (2000-2023)')
            plt.xlabel('Jahr')
            plt.ylabel('Anzahl der Nächtigungen')
            plt.grid(True)
            plt.xticks(jahre_numeric, rotation=45)  # Optional: Rotation der x-Achsen-Beschriftungen für bessere Lesbarkeit
            plt.tight_layout()
            plt.show()
