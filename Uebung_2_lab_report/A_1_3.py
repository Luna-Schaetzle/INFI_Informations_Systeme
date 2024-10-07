import pandas as pd
import matplotlib.pyplot as plt

# Lade die CSV-Datei
file_path = '/home/luna/5BHWII/INFI_Informations_Systeme/Uebung_2_lab_report/data/london weather.csv'
weather_data = pd.read_csv(file_path)

# Wir gehen ein Jahr durch und speichern uns die Mitteltemperaturen jedes Tages
# Als erstes gehen wir die Jahre durch und speichern uns die Mitteltemperaturen
years = weather_data['date'] // 10000
unique_years = years.unique()
all_mean_temps = []

for year in unique_years:
    mean_temps = weather_data[years == year]['mean_temp'].dropna().tolist()

    all_mean_temps.append(mean_temps)
    print(f"Im Jahr {year} war die durchschnittliche Temperatur {sum(mean_temps) / len(mean_temps)}°C")


# Nun erstellen wir ein Boxplot für alle Jahre
plt.boxplot(all_mean_temps, labels=unique_years)
plt.title("Temperaturen in London über die Jahre")
plt.xlabel("Jahr")
plt.ylabel("Temperature in °C")
plt.grid()
plt.savefig("A_1_3.png")
plt.show()

# Wir suchen uns nun die Extremwerte pro Jahr

# Wir suchen uns die Extremwerte pro Jahr
min_temps = []
max_temps = []
for year in unique_years:
    mean_temps = weather_data[years == year]['mean_temp'].tolist()
    min_temps.append(min(mean_temps))
    max_temps.append(max(mean_temps))
    #Nun schauen wir ob es einen Tag gibt, an dem die Temperatur extrem kalt oder warm war
    min_temp_day = weather_data[weather_data['mean_temp'] == min(mean_temps)]['date'].values[0]
    max_temp_day = weather_data[weather_data['mean_temp'] == max(mean_temps)]['date'].values[0]
    print(f"Im Jahr {year} war die kälteste Temperatur {min(mean_temps)}°C am {min_temp_day} und die wärmste Temperatur {max(mean_temps)}°C am {max_temp_day}")

# Nun erstellen wir ein Boxplot für alle Jahre
plt.plot(unique_years, min_temps, label="Minimale Temperatur pro Jahr", color="blue")
plt.plot(unique_years, max_temps, label="Maximale Temperatur pro Jahr", color="red")
plt.title("Extremtemperaturen in London über die Jahre")
plt.xlabel("Jahr")
plt.ylabel("Temperature in °C")
plt.grid()
plt.legend()
plt.savefig("A_1_3_extreme.png")
plt.show()


# Wir suchen uns nun noch extreme Schnewerte pro Jahr

# Wir suchen uns die Extremwerte pro Jahr
snow_dephts = []

# Wenn der Schneefallwert nicht vorhanden ist, dann ist der Wert NaN
# Wir ersetzen NaN durch 0
for year in unique_years:
    snow_depht = weather_data[years == year]['snow_depth'].fillna(0).tolist()
    snow_dephts.append(snow_depht)
    print(f"Im Jahr {year} war die durchschnittliche Schneehöhe {sum(snow_depht) / len(snow_depht)}cm")

# Nun erstellen wir ein Boxplot für alle Jahre
plt.boxplot(snow_dephts, labels=unique_years)
plt.title("Schneehöhe in London über die Jahre")
plt.xlabel("Jahr")
plt.ylabel("Schneehöhe in cm")
plt.grid()
plt.savefig("A_1_3_snow.png")
plt.show()
