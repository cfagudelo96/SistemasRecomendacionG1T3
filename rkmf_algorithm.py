"""Este módulo contiene la implementación de RKMF
    con Kernel lineal y non-negative constraints
    utilizada en el taller
"""
import math
import numpy as np
from surprise import AlgoBase
from surprise import PredictionImpossible

class RKMFAlgorithm(AlgoBase):
    """Clase que implementa RKMF con Kernel lineal y non-negative constraints"""
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02, noise=0.01):
        """Se inicializa el algoritmo con los hiper-parámetros dados por parámetro.
        Se inicializan adicionalmente los demás factores a utilizar en RKMF con valores temporales
        Parámetros:
            n_factors: Número de factores que tendrán las matrices de usuarios e ítems.
            n_epochs: Número de epochs en los que se aplicará el Stochastic Gradient Descent.
            lr: El learning rate del modelo.
            reg: El valor de regularización del modelo.
            noise: Valor que se utiliza a la hora de inicializar las matrices de usuarios x factores
                e ítems x factores.
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.noise = noise

        # Se inicializan los distintos factores relevantes en el algoritmo
        # con valores temporales
        self.init_low = 0.0
        self.init_high = 5.0
        self.kernel_a = 3.0
        self.kernel_c = 1
        self.pu = None
        self.qi = None
        AlgoBase.__init__(self)


    def fit(self, trainset):
        """Entrena el modelo con la información dada en un dataset de entrenamiento"""
        AlgoBase.fit(self, trainset)
        self.define_trainset_derived_attributes(trainset)
        self.sgd(trainset)
        return self


    def define_trainset_derived_attributes(self, trainset):
        """Define los valores de los factores relevantes para el 
            modelo a los que se les asignaron valores temporales
        """
        # Se calcula el rating mínimo y el rating máximo para poder definir
        # el valor de a, c y el rango de valores donde se inicializarán los
        # valores de la matriz de usuarios x factores e ítems x factores
        r_min = None
        r_max = None
        for _, _, rating in trainset.all_ratings():
            if r_min is None or rating < r_min:
                r_min = rating
            if r_max is None or rating > r_max:
                r_max = rating
        # Se definen a y c siguiendo la recomendación dada en el paper
        # Estos valores son usados para realizar la predicción de ratings
        self.kernel_a = r_min
        self.kernel_c = r_max - r_min
        # Se calcula el rango de valores donde se inicializarán los valores de las matrices
        # teniendo en cuenta la recomendación dada en el paper y el noise dado por parámetro
        k = self.n_factors
        g = trainset.global_mean
        init_base = math.sqrt((g - r_min)/(k * (r_max - r_min))) + self.noise
        self.init_low = init_base - self.noise
        if self.init_low < 0:
            self.init_low = 0
        self.init_high = init_base + self.noise


    def sgd(self, trainset):
        """Implementación de SGD para realizar la factorización de la matriz de utilidad"""
        rng = np.random.mtrand._rand
        # Se inicializa la matriz de usuarios y la matriz de ítems con valores aleatorios
        # en el intervalo definido anteriormente
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


    def user_update(self, user_id, item_id, rating):
        """Actualiza el vector de un usuario al recibir la información de un rating nuevo"""
        # Se obtiene el usuario y el ítem del trainset
        user = self.get_user(user_id)
        item = self.get_item(item_id)
        # Se añade el nuevo rating al trainset
        self.add_rating(user, item, rating)
        # Se obtienen los ratings que ha dado el usuario hasta el momento
        user_ratings = self.trainset.ur[user]

        # El vector del usuario en la matriz de usuarios x factores se reinicializa
        rng = np.random.mtrand._rand
        pu_user = rng.uniform(self.init_low, self.init_high, size=(self.n_factors))
        for epoch in range(self.n_epochs):
            # Se ajustan del vector del usuario el número de epochs definidos por parámetro
            print("Procesando epoch #" + str(epoch))
            for item_loop, rating_loop in user_ratings:
                # Se hace SGD por cada rating que el usuario ha dado
                # Se calcula el producto punto entre el vector
                # correspondiente al usuario e ítem actual en la iteración
                dot = np.dot(pu_user, self.qi[item_loop])
                # Se calcula el estimado actual para
                # el rating del usuario al ítem a + c * K(w_u, h_i)
                estimate = self.kernel_a + self.kernel_c * dot
                for factor in range(self.n_factors):
                    # Se calcula la derivada parcial con respecto a w_u,f
                    pd_pu = (estimate - rating_loop) * self.qi[item_loop, factor] + self.reg * pu_user[factor]
                    # Se calcula la derivada parcial con respecto a h_i,f
                    pd_qi = (estimate - rating_loop) * pu_user[factor] + self.reg * self.qi[item_loop, factor]

                    # Se actualiza el valor de w_u,f con non-negative restriction
                    pu_user[factor] = max(0, pu_user[factor] - self.lr * pd_pu)
        # Se actualiza el vector del usuario en la matriz
        self.pu[user] = pu_user


    def get_user(self, user_id):
        """Obtiene el inner_id del usuario dado como parámetro"""
        try:
            # Se trata de buscar el inner_id del usuario en el trainset
            user = self.trainset.to_inner_uid(str(user_id))
            return user
        except ValueError:
            # Si el usuario no tiene inner_id se le asigna el que seguiría,
            # es decir, el número de usuarios ya en el trainset
            user = self.trainset.n_users
            # Se asigna al raw id el inner id correspondiente
            self.trainset._raw2inner_id_users[user_id] = user
            # Se aumenta el número de usuarios en el trainset
            self.trainset.n_users += 1
            # Se crea un vector para el usuario y se agrega a la matriz de usuarios x factores
            rng = np.random.mtrand._rand
            pu_user = rng.uniform(self.init_low, self.init_high, size=(self.n_factors))
            self.pu = np.vstack((self.pu, pu_user))
            return user


    def get_item(self, item_id):
        """Obtiene el inner_id del ítem dado como parámetro"""
        try:
            # Se trata de buscar el inner_id del ítem en el trainset
            item = self.trainset.to_inner_iid(str(item_id))
            return item
        except ValueError:
            # Si el ítem no tiene inner_id se le asigna el que seguiría,
            # es decir, el número de ítems ya en el trainset
            item = self.trainset.n_items
            # Se asigna al raw id el inner id correspondiente
            self.trainset._raw2inner_id_items[item_id] = item
            # Se aumenta el número de ítems en el trainset
            self.trainset.n_items += 1
            # Se crea un vector para el ítem y se agrega a la matriz de ítems x factores
            rng = np.random.mtrand._rand
            qi_item = rng.uniform(self.init_low, self.init_high, size=(self.n_factors))
            self.qi = np.vstack((self.qi, qi_item))
            return item

    
    def add_rating(self, user, item, rating):
        """Añade un nuevo rating de un usuario a un ítem al training set"""
        # Se añade el rating a la lista de ratings del usuario
        self.trainset.ur[user].append((item, rating))
        # Se añade el rating a la lista de ratings del ítem
        self.trainset.ir[item].append((user, rating))
        # Se aumenta la cantidad de ratings en 1
        self.trainset.n_ratings += 1
