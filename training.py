import os
import math
from surprise import Dataset
from surprise import Reader

from rkmf_algorithm import RKMFAlgorithm


path = os.path.expanduser('/Users/cfagudelo/Desktop/trainset.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale=(0.5, 5.0))
data = Dataset.load_from_file(path, reader=reader)

trainset = data.build_full_trainset()

algo = RKMFAlgorithm()
algo.fit(trainset)

with open('/Users/cfagudelo/Desktop/validationset.csv') as file:
    file.readline()
    for line in file:
        line_split = line.split(',')
        user_id = line_split[0]
        item_id = line_split[1]
        rating = float(line_split[2])
        algo.user_update(user_id, item_id, rating)

with open('/Users/cfagudelo/Desktop/testset.csv') as file:
    file.readline()
    error_abs = 0
    error_sqr = 0
    n = 0
    for line in file:
        line_split = line.split(',')
        user_id = line_split[0]
        item_id = line_split[1]
        rating = float(line_split[2])
        rating_est = algo.predict(user_id, item_id).est
        error_abs += abs(rating - rating_est)
        error_sqr += (rating - rating_est)**2
        n += 1
    mae = error_abs / n
    rmse = math.sqrt(error_sqr / n)
    with open('./data/result.txt', 'w') as wfile:
        wfile.write('MAE: ' + str(mae) + '\n')
        wfile.write('RMSE: ' + str(rmse) + '\n')
