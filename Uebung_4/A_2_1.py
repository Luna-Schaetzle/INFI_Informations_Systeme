import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

# Zeichne die Regressionslinie
plt.plot(years, predictions, color='red', label='Regression Line')

# Beschriftungen und Titel
plt.xlabel('Year')
plt.ylabel('Total Population')
plt.title('Population Development in Tirol')
plt.legend()
plt.show()
