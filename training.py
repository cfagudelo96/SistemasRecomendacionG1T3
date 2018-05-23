import os
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import GridSearchCV

from rkmf_algorithm import RKMFAlgorithm


path = os.path.expanduser('/Users/cfagudelo/Desktop/ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale=(0.5, 5.0))
data = Dataset.load_from_file(path, reader=reader)

param_grid = {'n_factors': [50], 'n_epochs': [20], 'lr': [0.002], 'reg': [0.4]}

gs = GridSearchCV(RKMFAlgorithm, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
print(gs.best_score['mae'])
print(gs.best_params['mae'])
