import random

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

import config
from tuner import SVM_TUNER
import numpy as np

import logging
logger = None

def tune_with_flash(x_train, y_train, x_tune, y_tune, project_name, random_seed=0, ):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    logger = logging.getLogger(project_name)

    tuner = SVM_TUNER(random_seed)
    random.seed(random_seed)
    this_budget = config.BUDGET

    # Make initial population
    param_search_space = tuner.generate_param_pools(config.POOL_SIZE)

    # Evaluate initial pool
    evaluted_configs = random.sample(param_search_space, config.INIT_POOL_SIZE)
    param_search_space = list(set(param_search_space).difference(set(evaluted_configs)))

    f_scores = [measure_fitness(x_train, y_train, x_tune, y_tune, configs) for configs in evaluted_configs]

    # Filtering NaN case
    evaluted_configs, f_scores = filter_no_info(project_name, evaluted_configs, f_scores)


    logger.info(project_name + " | F Score of init pool: " + str(f_scores))

    # hold best values
    ids = np.argsort(f_scores)[::-1][:1]
    best_f = f_scores[ids[0]]
    best_config = evaluted_configs[ids[0]]

    # converting str value to int for CART to work
    evaluted_configs = [(x[0], tuner.label_transform(x[1]), x[2], x[3]) for x in evaluted_configs]
    param_search_space = [(x[0], tuner.label_transform(x[1]), x[2], x[3]) for x in param_search_space]

    # number of eval
    eval = 0
    while this_budget > 0:
        cart_model = DecisionTreeRegressor(random_state=1)

        cart_model.fit(evaluted_configs, f_scores)

        next_config_id = acquisition_fn(param_search_space, cart_model)
        next_config = param_search_space.pop(next_config_id)

        next_config_normal = (next_config[0], tuner.label_reverse_transform(next_config[1]), next_config[2], next_config[3])

        next_f = measure_fitness(x_train, y_train, x_tune, y_tune, next_config_normal)

        if np.isnan(next_f) or next_f == 0:
            continue

        f_scores.append(next_f)
        evaluted_configs.append(next_config)

        if isBetter(next_f, best_f):
            best_config = next_config_normal
            best_f = next_f
            this_budget += 1
            logger.info(project_name + " | new F: " + str(best_f) + " budget " + str(this_budget))
        this_budget -= 1
        eval += 1

    logger.info(project_name + " | Eval: " + str(eval))

    return best_config


def acquisition_fn(search_space, cart_model):
    predicted = cart_model.predict(search_space)

    ids = np.argsort(predicted)[::-1][:1]
    val = predicted[ids[0]]
    return ids[0]

def isBetter(new, old):
    return old < new

def measure_fitness(x_train, y_train, x_tune, y_tune, configs):
    clf = SVC(C=configs[0], kernel=configs[1], gamma=configs[2], coef0=configs[3], random_state=0)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_tune)
    cmat = confusion_matrix(y_tune, y_pred)

    return calc_f(cmat)


def calc_f(cmat):
    # Precision
    # ---------
    prec = cmat[1, 1] / (cmat[1, 1] + cmat[0, 1])

    # Recall
    # ------
    recall = cmat[1, 1] / (cmat[1, 1] + cmat[1, 0])

    # F1 Score
    # --------
    f1 = 2 * (prec * recall) / (prec + recall)

    return f1


def filter_no_info(label, evaluated_configs, fscores):
    for i, score in enumerate(fscores):
        if np.isnan(score) or score == 0:
            del evaluated_configs[i]
            del fscores[i]
            logger.info(label + "| filtered one: " + str(fscores))

    return evaluated_configs, fscores