# Importieren der notwendigen Bibliotheken
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Daten einlesen und aufbereiten

# Pfad zur Excel-Datei (bitte anpassen)
dateipfad = 'Zeitreihe-Winter-2024011810.xlsx'

# Lesen der Excel-Datei, überspringen der Zeilen 0 und 2, und setzen von Zeile 1 als Header
# Zeile 0 und 2 enthalten nur NaN-Werte oder irrelevante Informationen
df = pd.read_excel(dateipfad, skiprows=[0, 2], header=0)

# Manuelles Benennen der Spalten
# Annahme: Die ersten drei Spalten sind 'Bez', 'Gemnr', 'Gemeinde'
# Die restlichen Spalten entsprechen den Jahren von 2000 bis 2023
jahre = list(range(2000, 2024))
neue_spalten = ['Bez', 'Gemnr', 'Gemeinde'] + jahre
df.columns = neue_spalten

# Überprüfen der umbenannten Spalten
print("\nUmbenannte Spalten:")
print(df.columns)

# 2. Filtern der Daten für Innsbruck

# Filtern der Zeile für Innsbruck
# Annahme: 'Bez' == 'I' und 'Gemeinde' == 'Innsbruck' identifizieren eindeutig Innsbruck
innsbruck_df = df[(df['Bez'].str.strip() == 'I') & (df['Gemeinde'].str.strip() == 'Innsbruck')]

# Überprüfen, ob Innsbruck gefunden wurde
if innsbruck_df.empty:
    print("\nFehler: Die Gemeinde 'Innsbruck' wurde im Datensatz nicht gefunden.")
else:
    # Anzeigen der gefilterten Daten zur Überprüfung
    print("\nDaten für Innsbruck:")
    print(innsbruck_df[['Bez', 'Gemnr', 'Gemeinde'] + jahre].to_string(index=False))
    
    # 3. Extrahieren der Jahreswerte

    # Extrahieren der Jahreswerte als Liste
    labels = jahre
    values = innsbruck_df[jahre].values.flatten()

    # Überprüfen der extrahierten Werte
    print("\nJahreswerte für Innsbruck:")
    for jahr, anzahl in zip(labels, values):
        print(f"{jahr}: {int(anzahl)} Nächtigungen")

    # 4. Erstellen des Barplots mit Seaborn

    # Initialisieren des Plots
    plt.figure(figsize=(14, 8))

    # Erstellen des Barplots
    sns.barplot(x=labels, y=values, palette='terrain')

    # Titel und Beschriftungen hinzufügen
    plt.title('Jahreswerte der Nächtigungen in Innsbruck (2000-2023)', fontsize=16)
    plt.xlabel('Jahr', fontsize=14)
    plt.ylabel('Anzahl der Nächtigungen', fontsize=14)

    # Rotieren der x-Achsenbeschriftungen für bessere Lesbarkeit
    plt.xticks(rotation=70)

    # Anpassen des Layouts für bessere Darstellung
    plt.tight_layout()

    # Anzeigen des Plots
    plt.show()
