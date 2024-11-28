import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Lade den Datensatz
file_path = 'bev_meld.csv'
data = pd.read_csv(file_path)

# Entferne führende und nachfolgende Leerzeichen in der Spalte 'Gemeinde'
data['Gemeinde'] = data['Gemeinde'].str.strip()

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
innsbruck_population = innsbruck_data.iloc[0][population_columns].astype(float)

# Plot der Bevölkerungsdaten
plt.figure(figsize=(12, 6))
plt.plot(years, innsbruck_population, 'o-', label='Innsbruck Population', markersize=5)

# Regressionsanalyse
X = sm.add_constant(years)  # Füge eine Konstante für die Regression hinzu
model = sm.OLS(innsbruck_population, X).fit()  # Ordinary Least Squares Regression
predictions = model.predict(X)  # Berechnete Werte

# Funktion zur Bevölkerungsprognose
def population_forecast(years_to_predict, model):
    # Konvertiere in ein numpy Array, um mathematische Operationen auszuführen
    years_to_predict = np.array(years_to_predict)
    
    # Hole die Regressionskoeffizienten
    intercept, slope = model.params  # 'a' ist Steigung, 'b' ist Achsenabschnitt
    
    # Berechne die Prognosen für jedes Jahr
    predictions = intercept + slope * years_to_predict
    
    return predictions

# Prognose für das Jahr 2022 direkt mit den Koeffizienten
future_years_direct = [2022]
population_2022_direct = population_forecast(future_years_direct, model)

# Prognose für die Jahre 2022-2100 mit dem predict-Befehl
future_years_range = list(range(2022, 2101))
X_future = sm.add_constant(future_years_range)  # Füge eine Konstante für die zukünftigen Jahre hinzu
population_2022_2100_predict = model.predict(X_future)

# Ergebnisse anzeigen
print("Prognose der Bevölkerung für 2022 (direkt):", population_2022_direct[0])
print("Prognose der Bevölkerung für 2022-2100 mit predict:")
print(population_2022_2100_predict)

# Zeichne die Regressionslinie und die Prognosen
plt.plot(years, innsbruck_population, 'o-', label='Innsbruck Population', markersize=5)
plt.plot(years, predictions, color='red', label='Regression Line')
plt.plot(future_years_range, population_2022_2100_predict, 'o', color='blue', label='Population Forecast 2022-2100')
plt.axvline(x=2022, color='gray', linestyle='--', label='2022')

# Beschriftungen und Titel
plt.xlabel('Year')
plt.ylabel('Population of Innsbruck')
plt.title('Population Development and Forecast in Innsbruck')
plt.legend()
plt.show()
