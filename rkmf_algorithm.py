import math
import numpy as np

from surprise import AlgoBase
from surprise import PredictionImpossible

class RKMFAlgorithm(AlgoBase):
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02, noise=0.2):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.noise = noise
        self.init_low = 0.0
        self.init_high = 5.0
        self.bias = 3.0
        self.kernel_a = 3.0
        self.kernel_c = 1
        self.pu = None
        self.qi = None
        AlgoBase.__init__(self)


    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.define_trainset_derived_attributes(trainset)
        self.sgd(trainset)
        return self


    def define_trainset_derived_attributes(self, trainset):
        r_min = None
        r_max = None
        for _, _, rating in trainset.all_ratings():
            if r_min is None or rating < r_min:
                r_min = rating
            if r_max is None or rating > r_max:
                r_max = rating
        k = self.n_factors
        g = trainset.global_mean
        self.bias = trainset.global_mean
        self.kernel_a = r_min
        self.kernel_c = r_max - r_min
        init_base = math.sqrt((g - r_min)/(k * (r_max - r_min))) + self.noise
        self.init_low = init_base - self.noise
        if self.init_low < 0:
            self.init_low = 0
        self.init_high = init_base + self.noise


    def sgd(self, trainset):
        rng = np.random.mtrand._rand
        # Se inicializa la matriz de usuarios y la matriz de ítems con valores aleatorios
        pu = rng.uniform(self.init_low, self.init_high, size=(trainset.n_users, self.n_factors))
        qi = rng.uniform(self.init_low, self.init_high, size=(trainset.n_items, self.n_factors))
        for epoch in range(self.n_epochs):
            print("Procesando epoch #" + str(epoch))
            # Se ajustan los valores de la matriz el número de epochs definidos por parámetro
            for user, item, rating in trainset.all_ratings():
                # Se calcula el producto punto entre el vector
                # correspondiente al usuario e ítem actual en la iteración
                dot = np.dot(pu[user], qi[item])
                # Se calcula el estimado actual para
                # el rating del usuario al ítem a + c * K(w_u, h_i)
                estimate = self.kernel_a + self.kernel_c * dot
                for factor in range(self.n_factors):
                    # Se calcula la derivada parcial con respecto a w_u,f
                    pd_pu = (estimate - rating) * qi[item, factor] + self.reg * pu[user, factor]
                    # Se calcula la derivada parcial con respecto a h_i,f
                    pd_qi = (estimate - rating) * pu[user, factor] + self.reg * qi[item, factor]

                    # Se actualiza el valor de w_u,f y h_i,f con non-negative restriction
                    pu[user, factor] = max(0, pu[user, factor] - self.lr * pd_pu)
                    qi[item, factor] = max(0, qi[item, factor] - self.lr * pd_qi)
        self.pu = pu
        self.qi = qi


    def dot_product(self, pu, qi, user, item):
        """Calcula el producto punto entre un vector de usuario y un vector de ítem"""
        dot = 0
        for factor in range(self.n_factors):
            dot += pu[user, factor] * qi[item, factor]
        return dot


    def estimate(self, user, item):
        """Estima el rating que un usuario dará a un ítem"""
        known_user = self.trainset.knows_user(user)
        known_item = self.trainset.knows_item(item)
        est = self.trainset.global_mean
        if known_user and known_item:
            est = self.kernel_a + self.kernel_c * np.dot(self.qi[item], self.pu[user])
        else:
            raise PredictionImpossible("User and item are unknown.")
        return est
