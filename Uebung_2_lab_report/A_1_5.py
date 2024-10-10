import pandas as pd
import matplotlib.pyplot as plt

# Lade die CSV-Datei
file_path = '/home/luna/5BHWII/INFI_Informations_Systeme/Uebung_2_lab_report/data/london weather.csv'
weather_data = pd.read_csv(file_path)

# Extrahiere die Jahre aus der Datenspalte
weather_data['year'] = weather_data['date'] // 10000
unique_years = weather_data['year'].unique()

# Berechne den durchschnittlichen Niederschlag pro Jahr
avg_precipitation = []
for year in unique_years:
    avg_precipitation.append(weather_data[weather_data['year'] == year]['precipitation'].dropna().mean())

# Erstelle ein Liniendiagramm für den durchschnittlichen Niederschlag pro Jahr
plt.plot(unique_years, avg_precipitation)
plt.title("Durchschnittliche Niederschläge der letzten Jahre in London")
plt.xlabel("Jahr")
plt.ylabel("Durchschnittlicher Niederschlag in mm")
plt.grid()
plt.savefig("A_1_5.png")
plt.show()
