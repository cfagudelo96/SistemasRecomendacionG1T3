from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split

from rkmf_algorithm import RKMFAlgorithm


data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=0.90)
algo = RKMFAlgorithm()
algo.fit(trainset)
for user_id, item_id, rating in testset:
    algo.user_update(user_id, item_id, rating)
