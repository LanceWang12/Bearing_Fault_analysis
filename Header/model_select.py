from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

from Header.metric import modelreport

def tuning_param(alg, parameter, name_lst, x_train, y_train, x_test, y_test, test_time = 5):
    # Tuning param
    acc_lst = []
    param = list(parameter.values())[0]
    for i in range(test_time):
        acc = []
        for j in param:
            model = eval(alg)
            model.fit(x_train, y_train)
            tmp, _ = modelreport(model, x_test, y_test, target_names = name_lst, mtype='ML', verbose = False)
            acc.append(tmp)
        acc_lst.append(acc)
        

    mean_acc = np.mean(np.array(acc_lst), axis = 0)

    print("{} report:".format(list(parameter.keys())[0]))
    print('The most approriate value is {} (by Accuracy).'.format(param[np.argmax(mean_acc)]))

    plt.figure(figsize = (10, 7))
    plt.plot(param, mean_acc, label = 'Accuracy')
    plt.legend(prop={'size': 15})
    plt.show()
    
    return model

def GetFeatureImportanceAndMask(alg, x, y, percentage = 0.8):
    # Get Mean Decease Impurity
    mdi = alg.feature_importances_
    # Get Permutation Importance
    pi = permutation_importance(alg, x, y, n_repeats=30, random_state=0).importances_mean

    # Normalize
    mdi = (mdi - np.mean(mdi)) / np.std(mdi)
    pi = (pi - np.mean(pi)) / np.std(pi)

    total_importance = mdi + pi

    mask = np.argsort(total_importance)[::-1]

    return total_importance, mask