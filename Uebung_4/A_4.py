import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Lade den Datensatz
file_path = 'bev_meld.csv'
data = pd.read_csv(file_path)

# Entferne führende und nachfolgende Leerzeichen in der Spalte 'Bezirk'
data['Bezirk'] = data['Bezirk'].str.strip()

# Filtere die Daten für die beiden Bezirke (IL und RE)
bezirk_1 = 'IL'
bezirk_2 = 'KU'

data_bezirk_1 = data[data['Bezirk'] == bezirk_1]
data_bezirk_2 = data[data['Bezirk'] == bezirk_2]

# Aggregiere die Gesamtbevölkerung für jeden Bezirk pro Jahr
population_columns = data.columns[3:]  # Spalten ab 1993
years = population_columns.astype(int)

total_population_bezirk_1 = data_bezirk_1[population_columns].sum()
total_population_bezirk_2 = data_bezirk_2[population_columns].sum()

# Regressionsanalyse für beide Bezirke
X = sm.add_constant(years.astype(int))  # Jahre als Prädiktor hinzufügen

model_bezirk_1 = sm.OLS(total_population_bezirk_1, X).fit()
model_bezirk_2 = sm.OLS(total_population_bezirk_2, X).fit()

# Vorhersagen basierend auf den Modellen
predictions_bezirk_1 = model_bezirk_1.predict(X)
predictions_bezirk_2 = model_bezirk_2.predict(X)

# Extrahiere die Steigungen (Coefficients)
slope_bezirk_1 = model_bezirk_1.params[1]
slope_bezirk_2 = model_bezirk_2.params[1]

# Plot: Vergleich der Bevölkerungsentwicklung
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Bevölkerung Bezirk 1
axes[0, 0].plot(years, total_population_bezirk_1, 'o-', label=f'{bezirk_1} Population')
axes[0, 0].plot(years, predictions_bezirk_1, color='red', label='Regression Line')
axes[0, 0].set_title(f'Population Development in {bezirk_1}')
axes[0, 0].set_xlim([1993, 2030])
axes[0, 0].legend()

# Bevölkerung Bezirk 2
axes[0, 1].plot(years, total_population_bezirk_2, 'o-', label=f'{bezirk_2} Population')
axes[0, 1].plot(years, predictions_bezirk_2, color='red', label='Regression Line')
axes[0, 1].set_title(f'Population Development in {bezirk_2}')
axes[0, 1].set_xlim([1993, 2030])
axes[0, 1].legend()

# Steigung von Bezirk 1
axes[1, 0].bar([bezirk_1], [slope_bezirk_1], color='blue')
axes[1, 0].set_title(f'Regression Slope for {bezirk_1}')
axes[1, 0].set_ylim([0, max(slope_bezirk_1, slope_bezirk_2) * 1.2])

# Steigung von Bezirk 2
axes[1, 1].bar([bezirk_2], [slope_bezirk_2], color='green')
axes[1, 1].set_title(f'Regression Slope for {bezirk_2}')
axes[1, 1].set_ylim([0, max(slope_bezirk_1, slope_bezirk_2) * 1.2])

# Layout und Anzeige
fig.tight_layout()
plt.show()
