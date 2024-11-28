import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Lade den Datensatz
file_path = 'bev_meld.csv'
data = pd.read_csv(file_path)

# Entferne führende und nachfolgende Leerzeichen in der Spalte 'Gemeinde'
data['Gemeinde'] = data['Gemeinde'].str.strip()

# Debugging: Überprüfe, welche Gemeinden in der Spalte 'Gemeinde' vorhanden sind
print("Verfügbare Gemeinden:", data['Gemeinde'].unique())

# Filtere die Daten nur für Innsbruck
innsbruck_data = data[data['Gemeinde'] == 'Innsbruck']

# Überprüfe, ob Daten für Innsbruck gefunden wurden
if innsbruck_data.empty:
    print("Fehler: Keine Daten für 'Innsbruck' gefunden. Überprüfe die CSV-Datei.")
    exit()

# Definiere die Bevölkerungs-Spalten (von 1993 bis 2021)
population_columns = innsbruck_data.columns[3:]  # Die Spalten ab 1993

# Extrahiere die Jahre und die Bevölkerung von Innsbruck
years = population_columns.astype(int)
innsbruck_population = innsbruck_data.iloc[0][population_columns].astype(float)  # Bevölkerung von Innsbruck

# Plot der Bevölkerungsdaten
plt.figure(figsize=(12, 6))
plt.plot(years, innsbruck_population, 'o-', label='Innsbruck Population', markersize=5)

# Regressionsanalyse
X = sm.add_constant(years)  # Füge eine Konstante für die Regression hinzu
model = sm.OLS(innsbruck_population, X).fit()  # Ordinary Least Squares Regression
predictions = model.predict(X)  # Berechnete Werte

# Zeichne die Regressionslinie
plt.plot(years, predictions, color='red', label='Regression Line')

# Beschriftungen und Titel
plt.xlabel('Year')
plt.ylabel('Population of Innsbruck')
plt.title('Population Development in Innsbruck')
plt.legend()
plt.show()
