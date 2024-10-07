import pandas as pd
import matplotlib.pyplot as plt

# Lade die CSV-Datei
file_path = '/home/luna/5BHWII/INFI_Informations_Systeme/Uebung_2_lab_report/data/london weather.csv'
weather_data = pd.read_csv(file_path)

# Wir gehen ein Jahr durch und speichern uns die Mitteltemperaturen jedes Tages
# Als erstes gehen wir die Jahre durch und speichern uns die Mitteltemperaturen
years = weather_data['date'] // 10000
unique_years = years.unique()

# entferne alle Jahre die nicht in den letzten 10 Jahren liegen
unique_years = unique_years[-10:]


# Wir suchen uns die Mittelwerte der letzten 10 Jahre
mean_temps = []
for year in unique_years:
    mean_temp = weather_data[years == year]['mean_temp'].dropna().tolist()
    mean_temps.append(sum(mean_temp) / len(mean_temp))

# Nun erstellen wir ein Balkendiagramm für die Mitteltemperaturen der letzten 10 Jahre
plt.bar(unique_years, mean_temps)
plt.title("Mitteltemperaturen der letzten 10 Jahre in London")
plt.xlabel("Jahr")
plt.ylabel("Temperature in °C")
plt.grid()
plt.savefig("A_1_4.png")
plt.show()
