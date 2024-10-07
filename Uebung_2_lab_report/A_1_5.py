

import pandas as pd
import matplotlib.pyplot as plt

# Lade die CSV-Datei
file_path = '/home/luna/5BHWII/INFI_Informations_Systeme/Uebung_2_lab_report/data/london weather.csv'
weather_data = pd.read_csv(file_path)

# Wir gehen ein Jahr durch und speichern uns die Mitteltemperaturen jedes Tages
# Als erstes gehen wir die Jahre durch und speichern uns die Mitteltemperaturen
years = weather_data['date'] // 10000
unique_years = years.unique()

# Wir untersuchen die veränderung des niederschlages "percipitation" über die Jahre
percipitation = []
for year in unique_years:
    percipitation.append(weather_data[years == year]['precipitation'].dropna().tolist())

# Nun erstellen wir ein lienien diagramm für die Niederschläge der letzten 10 Jahre
plt.plot(unique_years, percipitation)
plt.title("Niederschläge der letzten 10 Jahre in London")
plt.xlabel("Jahr")
plt.ylabel("Niederschlag in mm")
plt.grid()
plt.savefig("A_1_5.png")
plt.show()
