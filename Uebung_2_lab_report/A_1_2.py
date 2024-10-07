# Eine Temperaturkurve für ein beliebiges Jahr soll erstellt werden, vorzugsweise als Punktdiagramm

import numpy as np
from matplotlib import pyplot as plt

#source: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data
d = np.genfromtxt('/home/luna/5BHWII/INFI_Informations_Systeme/Uebung_2_lab_report/data/london weather.csv', delimiter=",", skip_header=1 )

print("Welches Jahr soll dargestellt werden?")
year_input = int(input())

# Interpretation der Daten in die Grafik einbauen
print("Wir sehen einen eindeutigen Jahreszeitenverlauf, wobei die Temperaturen im Sommer am höchsten sind und im Winter am niedrigsten ist.")

plt.title("Mean temperature of every day in the year " + str(year_input))
plt.xlabel("Day")
plt.ylabel("Mean Temperature")
day = 0
# mean temperature of every day in the year in a point diagram
for i in range(1, len(d)):
    year = (d[i,0] % 100000000 / 10000).astype('i')
    if year == year_input:
        day = day + 1
        mean_temp = d[i, 6]
        print("Day:", day, "Mean Temp:", mean_temp)
        plt.plot(day, mean_temp, "r.")

plt.savefig("A_1_2.png")
plt.show()

    