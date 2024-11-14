import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Lade den Datensatz
file_path = 'bev_meld.csv'
data = pd.read_csv(file_path)

# Definiere die Bevölkerungs-Spalten (von 1993 bis 2021)
population_columns = data.columns[3:]  # Die Spalten ab 1993

# Berechne die Gesamtbevölkerung für jedes Jahr
data['Total_Population'] = data[population_columns].sum()

# Extrahiere die Jahre und die Gesamtbevölkerung pro Jahr
years = population_columns.astype(int)
total_population_per_year = data[population_columns].sum()

# Plot der Bevölkerungsdaten
plt.figure(figsize=(12, 6))
plt.plot(years, total_population_per_year, 'o-', label='Total Population', markersize=5)

# Regressionsanalyse
X = sm.add_constant(years)  # Füge eine Konstante für die Regression hinzu
model = sm.OLS(total_population_per_year, X).fit()  # Ordinary Least Squares Regression
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

# Prognose für das Jahr 2030 direkt mit den Koeffizienten
future_years_direct = [2022]
population_2030_direct = population_forecast(future_years_direct, model)

# Prognose für die Jahre 2030-2100 mit dem predict-Befehl
future_years_range = list(range(2022, 2101))
X_future = sm.add_constant(future_years_range)  # Füge eine Konstante für die zukünftigen Jahre hinzu
population_2030_2100_predict = model.predict(X_future)

# Ergebnisse anzeigen
print("Prognose der Bevölkerung für 2023-2100 mit predict:")
print(population_2030_2100_predict)


# Zeichne die Regressionslinie für die Zukunft

# Plot der Bevölkerungsdaten
plt.plot(years, total_population_per_year, 'o-', label='Total Population', markersize=5)

# Zeichne die Regressionslinie
plt.plot(years, predictions, color='red', label='Regression Line')

# Prognose für 2030
#plt.plot(future_years_direct, population_2030_direct, 'o', color='green', label='Population Forecast 2030')
plt.plot(future_years_range, population_2030_2100_predict, 'o', color='blue', label='Population Forecast 2022-2100')
plt.axvline(x=2022, color='gray', linestyle='--', label='2022')

# Beschriftungen und Titel
plt.xlabel('Year')
plt.ylabel('Total Population')
plt.title('Population Development in Tirol')
plt.legend()
plt.show()
# Output:
# Prognose der Bevölkerung für 2030 (direkt): 798662.5


