import pathlib
import pandas as pd

finetuning = False
group = True

testing_mode = 'group' if group else 'finetuning' if finetuning else 'complete'

results = pd.read_csv(pathlib.Path(f'src/../results/loso_results_{testing_mode}.csv'), index_col=0)

if not finetuning:
    total_scores = results.agg({'Accuracy Random Forest':['mean', 'std'], 'Accuracy CNN':['mean', 'std'], 'Accuracy RNN':['mean', 'std'], 'Accuracy CRNN':['mean', 'std'], 'Accuracy Feature RNN':['mean', 'std'], 'Accuracy Ensemble':['mean', 'std']})
    total_scores.to_csv(pathlib.Path(f'src/../results/total_results_{testing_mode}.csv'), index=False)

    group_scores = results.groupby(by='Group').agg({'Accuracy Random Forest':['mean', 'std'], 'Accuracy CNN':['mean', 'std'], 'Accuracy RNN':['mean', 'std'], 'Accuracy CRNN':['mean', 'std'], 'Accuracy Feature RNN':['mean', 'std'], 'Accuracy Ensemble':['mean', 'std']})
    group_scores.to_csv(pathlib.Path(f'src/../results/group_results_{testing_mode}.csv'), index=False)

else:
    total_scores = results.agg({'Accuracy CNN':['mean', 'std'], 'Accuracy RNN':['mean', 'std'], 'Accuracy CRNN':['mean', 'std'], 'Accuracy Feature RNN':['mean', 'std'], 'Accuracy Ensemble':['mean', 'std']})
    total_scores.to_csv(pathlib.Path(f'src/../results/total_results_{testing_mode}.csv'), index=False)

    group_scores = results.groupby(by='Group').agg({'Accuracy CNN':['mean', 'std'], 'Accuracy RNN':['mean', 'std'], 'Accuracy CRNN':['mean', 'std'], 'Accuracy Feature RNN':['mean', 'std'], 'Accuracy Ensemble':['mean', 'std']})
    group_scores.to_csv(pathlib.Path(f'src/../results/group_results_{testing_mode}.csv'), index=False)