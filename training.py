from surprise import Dataset
from surprise.model_selection import GridSearchCV

from rkmf_algorithm import RKMFAlgorithm


data = Dataset.load_builtin('ml-100k')

param_grid = {'noise': [0.01, 0.009, 0.2, 0.3]}

gs = GridSearchCV(RKMFAlgorithm, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# best RMSE score
print(gs.best_score['mae'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['mae'])
