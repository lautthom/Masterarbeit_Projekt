import pathlib
import pandas as pd

results_loso = pd.read_csv(pathlib.Path('src/../results/loso_results.csv'), index_col=0)

results = results_loso['Accuracy Ensemble']

print(results)

results = results.to_list()
print(results_loso)

results_male_young = results[0:15]
results_male_middle = results[15:30]
results_male_old = results[30:44]
results_female_young = results[44:59]
results_female_middle = results[59:74]
results_female_old = results[74:]

print(sum(results_male_young) / len(results_male_young))
print(sum(results_male_middle) / len(results_male_middle))
print(sum(results_male_old) / len(results_male_old))
print(sum(results_female_young) / len(results_female_young))
print(sum(results_female_middle) / len(results_female_middle))
print(sum(results_female_old) / len(results_female_old))

print(sum(results) / len(results))
