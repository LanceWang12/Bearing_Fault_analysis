from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


def modelreport(alg, X, Y, target_names, mtype='ML', verbose = True):
    # alg: A classifier
    # X: some features
    # Y: labels
    # target_names: class name(Ex.['inner_defect', ..., 'ball'])
    
    #Predict testing set:
    if mtype == 'DL':
        dtest_predprob = alg.predict(X)
        dtest_predictions = alg.predict_classes(X)
    elif mtype == 'ML':
        dtest_predictions = alg.predict(X)
        dtest_predprob = alg.predict_proba(X)

    #Print model report:
    acc = accuracy_score(Y, dtest_predictions)
    auc = roc_auc_score(Y, dtest_predprob, multi_class="ovr", average="weighted")
    if verbose:
        print ("Accuracy : %.4g" %acc )
        print(classification_report(Y, dtest_predictions, target_names = target_names))
        print ("AUC Score: %f" %auc )

    return acc, auc

