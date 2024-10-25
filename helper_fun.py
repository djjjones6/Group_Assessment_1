import math
from sklearn.metrics import confusion_matrix
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from helper_fun import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

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

def evaluating_performance_diff_splits(model,split_data, adjust_threshold=False,threshold=0.5,display_confusion=True,display_roc=True, key_metric_only=False):
    performance = {}
    if display_confusion == True:
        #Create a 2x2 grid for confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    if display_roc == True:
        plt.figure(figsize=(8,6))
        plt.figure()
    colors = ['darkorange', 'blue', 'green', 'red']
    # Loop through the splits in split_data
    for i, split in enumerate(split_data):
        X_train, X_val, y_train, y_val =split_data[split]
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict probabilities for the test set
        y_score = model.predict_proba(X_val)[:, 1]

        # Predict our target variable
        y_val_pred = model.predict(X_val)
        if adjust_threshold == True:
            y_val_pred = (y_score > threshold).astype(int)
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_val, y_score)
        roc_auc = auc(fpr, tpr)
        #Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
        # Generate classification report
        report = classification_report(y_val, y_val_pred)
        # Extract metrics from classification report
        report_dict = classification_report(y_val, y_val_pred, output_dict=True)
        balanced_MCC = balanced_MCC_macro_avg(y_val, y_val_pred)
        
        if key_metric_only == False:
            performance[split] = {
                'precision': report_dict['1']['precision'],
                'recall': report_dict['1']['recall'],
                'f1-score': report_dict['1']['f1-score'],
                'support': report_dict['1']['support'],
                'roc_auc': roc_auc,
                'balanced_accuracy': balanced_acc,
                'balanced_MCC': balanced_MCC,
            }
        else:
            performance[split] = {
                'recall' : report_dict['1']['recall'],
                'f1-score' : report_dict['1']['f1-score'],
                'balalanced_MCC' : balanced_MCC
            }
        
        
        #Plot the confusion matrix for the teset set 
        conf_matrix = confusion_matrix(y_val, y_val_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        #Normalize the confusion matrix by the number of true samples per class
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        if display_confusion == True:
            sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[i],
                        xticklabels=['Not Exited (0)', 'Exited (1)'],
                        yticklabels=['Not Exited (0)', 'Exited (1)'])
            axes[i].set_title(f'Confusion Matrix - {split} Split')
            axes[i].set_ylabel('Actual')
            axes[i].set_xlabel('Predicted')
            
        if display_roc == True:
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC curve {split} (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic for different splits')
            plt.legend(loc="lower right")
    plt.show()
    performance = pd.DataFrame(performance).T
    return performance