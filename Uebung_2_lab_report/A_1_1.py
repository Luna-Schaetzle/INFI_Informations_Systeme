import numpy as np
from matplotlib import pyplot as plt

#source: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data
d = np.genfromtxt('/home/luna/5BHWII/INFI_Informations_Systeme/Uebung_2_lab_report/data/london weather.csv', delimiter=",", skip_header=1 )

dt =  d[:,0] #Datum mit folgendem Aufbau: 19790103 (3.Jänner 1979)
# Aufteilen in Tag, Monat, Jahr
day = (dt % 100).astype('i')
month = (dt % 10000 / 100).astype('i')
year = (dt % 100000000 / 10000).astype('i')


# Darstellung der Temperaturunterschiede
# Aktuell interessant ist, wie sich die Temperatur über die Jahre verändert hat. zu diesem Zweck einfach 4
# aussagekräftige Jahre aussuchen und diese Unterschiede deutlich machen, am Besten als Boxplots.

mean_temp_1980 = np.mean(d[year == 1980, 6])
mean_temp_1990 = np.mean(d[year == 1990, 6])
mean_temp_2000 = np.mean(d[year == 2000, 6])
mean_temp_2010 = np.mean(d[year == 2010, 6])

print("Mean Temp 1980:", mean_temp_1980)
print("Mean Temp 1990:", mean_temp_1990)
print("Mean Temp 2000:", mean_temp_2000)
print("Mean Temp 2010:", mean_temp_2010)

# interpretation der Daten in die Grafik einbauen
print("Die durchschnittliche Temperatur der Jahre wird nicht nur wärmer, sondern auch extremer wie wir im Jahr 2010 sehen können.") 

# Gegenüberstellung der Mitteltemperaturen
plt.boxplot([d[year == 1980, 6], d[year == 1990, 6], d[year == 2000, 6], d[year == 2010, 6]])
plt.xticks([1,2,3,4], ["1980", "1990", "2000", "2010"])
plt.ylabel("Temperature in °C")
plt.title("Temperaturen in London")
plt.grid()
plt.savefig("/home/luna/5BHWII/INFI_Informations_Systeme/Uebung_2_lab_report/A_1_1.png")
plt.show()




