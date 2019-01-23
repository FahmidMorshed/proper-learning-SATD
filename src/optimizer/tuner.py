import random

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class DT_TUNER:
    def __init__(self, seed=0):
        self.criterion = ['gini', 'entropy']
        self.splitter = ['best', 'random']
        self.max_depth = [2, 1000]
        self.max_feature = [.1, 1]
        self.class_weight = [1, 9]
        random.seed(seed)

        encoder_criterion = LabelEncoder()
        self.encoder_criterion = encoder_criterion.fit(self.criterion)
        encoder_splitter = LabelEncoder()
        self.encoder_splitter = encoder_splitter.fit(self.splitter)

        self.default_config = ('gini', 'best', None, None, None)

    def generate_param_combinaions(self):
        criterion = random.choice(self.criterion)
        splitter = random.choice(self.splitter)
        max_depth = random.randint(self.max_depth[0], self.max_depth[1])
        max_feature = random.uniform(self.max_feature[0], self.max_feature[1])

        class_weight = random.randint(self.class_weight[0], self.class_weight[1])
        class_weight = {'yes': class_weight, 'no': (10-class_weight)}

        return criterion, splitter, max_depth, max_feature, class_weight

    def criterion_transform(self, val):
        arr_list = self.encoder_criterion.transform([val])
        return float(arr_list.tolist()[0])

    def criterion_reverse_transform(self, val):
        arr_list = self.encoder_criterion.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def splitter_transform(self, val):
        arr_list = self.encoder_splitter.transform([val])
        return float(arr_list.tolist()[0])

    def splitter_reverse_transform(self, val):
        arr_list = self.encoder_splitter.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def generate_param_pools(self, size):
        list_of_params = [self.generate_param_combinaions() for x in range(size)]
        return list_of_params

    def get_clf(self, configs):
        clf = DecisionTreeClassifier(criterion=configs[0], splitter=configs[1], max_depth=configs[2],
                                     max_features=configs[3],
                               class_weight=configs[4], random_state=0)
        return clf

    def transform_to_numeric(self, x):
        return self.criterion_transform(x[0]), self.splitter_transform(x[1]), x[2], x[3], x[4].get('yes')

    def reverse_transform_from_numeric(self, x):
        return self.criterion_reverse_transform(x[0]), self.splitter_reverse_transform(x[1]), x[2], x[3], \
               {'yes': x[4], 'no': (10-x[4])}


class SVM_TUNER:
    def __init__(self, fold_num=0):
        self.C_VALS = [1, 50]
        self.KERNELS = ['rbf', 'linear', 'sigmoid', 'poly']
        self.GAMMAS = [0, 1]
        self.COEF0S = [0, 1]
        self.enc = None
        random.seed(fold_num)

        self.label_coding()

    def generate_param_combinaions(self):
        c = random.uniform(self.C_VALS[0], self.C_VALS[1])
        kernel = random.choice(self.KERNELS)
        gamma = random.uniform(self.GAMMAS[0], self.GAMMAS[1])
        coef0 = random.uniform(self.COEF0S[0], self.COEF0S[1])

        return c, kernel, gamma, coef0

    def label_coding(self):
        enc = LabelEncoder()
        enc.fit(self.KERNELS)
        self.enc = enc

    def label_transform(self, val):
        arr_list = self.enc.transform([val])
        return float(arr_list.tolist()[0])

    def label_reverse_transform(self, val):
        arr_list = self.enc.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def generate_param_pools(self, size):
        list_of_params = [self.generate_param_combinaions() for x in range(size)]
        return list_of_params

