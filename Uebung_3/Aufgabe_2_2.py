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

# Lesen der Excel-Datei, überspringen der Zeilen 0 und 2, und setzen von Zeile 1 als Header
df = pd.read_excel(dateipfad, skiprows=[0, 2], header=0)

# Überprüfen, ob die Daten korrekt eingelesen wurden
print("Erste fünf Zeilen des eingelesenen Datensatzes:")
print(df.head())

# 1.3 Import und Kontrolle

# Manuelles Benennen der Spalten
# Die ersten drei Spalten sind 'Bez', 'Gemnr', 'Gemeinde'
# Die restlichen Spalten entsprechen den Jahren 2000 bis 2023

neue_spalten = ['Bez', 'Gemnr', 'Gemeinde'] + [f'x{jahr}' for jahr in range(2000, 2024)]
df.columns = neue_spalten

# Überprüfen der umbenannten Spalten
print("\nUmbenannte Spalten:")
print(df.columns)

# Kontrolle der Daten mit describe()
beschreibung = df.describe()

# Verwenden von tabulate für eine bessere Ausgabe
print("\nBeschreibende Statistik des Datensatzes:")
print(tabulate(beschreibung, headers='keys', tablefmt='psql'))

# Anzeigen der verfügbaren Bezirke im Datensatz
verfuegbare_bezirke = df[['Bez', 'Gemeinde']].drop_duplicates()
print("\nVerfügbare Bezirke im Datensatz:")
print(verfuegbare_bezirke)

# 2. Erste Auswertung

# 2.1 Wachstum darstellen für einen ausgewählten Bezirk

def wachstum_bezirk(df, bez_code, bez_name):
    """
    Analysiert und visualisiert das Wachstum des angegebenen Bezirks.

    Parameters:
    - df: pandas DataFrame, der den Datensatz enthält
    - bez_code: str, der Code des Bezirks (z.B. 'IM')
    - bez_name: str, der Name des Bezirks für die Anzeige
    """
    # Filtern der Zeilen für den angegebenen Bezirk
    bezirk_df = df[df['Bez'].str.strip() == bez_code]
    
    # Überprüfen, ob der Bezirk gefunden wurde
    if bezirk_df.empty:
        print(f"\nFehler: Der Bezirk mit Code '{bez_code}' wurde im Datensatz nicht gefunden.")
        return
    
    # Identifizieren der Jahr-Spalten im DataFrame
    jahres_spalten = [col for col in df.columns if col.startswith('x')]
    
    # Summieren der Nächtigungen über alle Gemeinden im Bezirk für jedes Jahr
    wachstum = bezirk_df[jahres_spalten].sum(axis=0)
    
    # Umwandeln der Spaltennamen zurück zu Jahreszahlen
    jahre = [int(col[1:]) for col in wachstum.index]
    
    # Erstellen eines Dictionaries für die Anzeige
    wachstum_dict = dict(zip(jahre, wachstum.values))
    
    # Ausgabe der Wachstumszahlen in der Konsole
    print(f"\nWachstum des Bezirks '{bez_name}' ({bez_code}) von 2000 bis 2023:")
    for jahr, anzahl in wachstum_dict.items():
        print(f"{jahr}: {int(anzahl)} Nächtigungen")
    
    # Plotten des Wachstums als Liniendiagramm
    plt.figure(figsize=(12, 6))
    plt.plot(jahre, wachstum.values, marker='o', linestyle='-', color='green')
    plt.title(f'Wachstum der Nächtigungen im Bezirk {bez_name} ({bez_code}) (2000-2023)')
    plt.xlabel('Jahr')
    plt.ylabel('Anzahl der Nächtigungen')
    plt.grid(True)
    plt.xticks(jahre, rotation=45)
    plt.tight_layout()
    plt.show()

# Beispielaufruf für den Bezirk 'IM' (z.B. 'Imst')
# Da 'IM' mehrere Gemeinden umfasst, ist es sinnvoll, einen repräsentativen Namen zu wählen, z.B. 'Imst Bezirk'
wachstum_bezirk(df, 'IM', 'Imst Bezirk')

# Hinweis: Wenn Sie einen anderen Bezirk analysieren möchten, ändern Sie die Parameter entsprechend.
# Beispiel:
# wachstum_bezirk(df, 'XYZ', 'Bezirksname')

