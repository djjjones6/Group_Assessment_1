import math
from sklearn.metrics import confusion_matrix
import numpy as np

def balanced_MCC_macro_avg(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    classes = C.shape[0]
    bal_MCC_each_class = []
    for i in range(classes):
        TP = C[i][i]
        FN = 0
        FP = 0
        TN = 0
        for j in range(classes):
            if j != i:
                FN = FN + C[i][j]
                FP = FP + C[j][i] 
                for k in range(classes):
                    if k != i:
                        TN = TN + C[j][k]
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        x = (sens + spec - 1)/(math.sqrt(1-(sens-spec)**2))
        bal_MCC_each_class.append(x)
    return np.mean(bal_MCC_each_class)