# It creates ROC and PRC for Chewing Damage. 
# Also, it reports other indices like F1-Score, etc.
import numpy as np
import pandas as pd
from numpy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pandas import DataFrame
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from prepare_data import working_data_dummies

#***** There is a restriction on the age. It must be less than 70
cover4_data=working_data_dummies[working_data_dummies.ALDER<70]

#**** Inherit everything in the new dataset
cover4_data_scaled=cover4_data.copy()
#*** Apply the MinMax scaling
cover4_data_scaled[['ALDER','BESKTITY']]=preprocessing.minmax_scale(cover4_data_scaled[['ALDER','BESKTITY']])

cols_for_testing=['ALDER','BESKTITY',
                  'TYPKODE_Factor2_2',
       'TYPKODE_Factor2_3', 'TYPKODE_Factor2_4', 'TYPKODE_Factor2_5',
       'TYPKODE_Factor2_6', 'TYPKODE_Factor2_7', 'TYPKODE_Factor2_8',
       'TYPKODE_Factor2_9', 'TYPKODE_Factor2_10', 'TYPKODE_Factor2_11',
       'TYPKODE_Factor2_12', 'TYPKODE_Factor2_13', 'TYPKODE_Factor2_14',
       'TYPKODE_Factor2_15', 'TYPKODE_Factor2_16', 'TYPKODE_Factor2_17',
       'TYPKODE_Factor2_18', 'TYPKODE_Factor2_19', 'TYPKODE_Factor2_20',
       'TYPKODE_Factor2_21', 'TYPKODE_Factor2_22', 'TYPKODE_Factor2_23',
       'TYPKODE_Factor2_24', 'TYPKODE_Factor2_25', 'TYPKODE_Factor2_26',
       'TYPKODE_Factor2_27', 'TYPKODE_Factor2_28', 'TYPKODE_Factor2_29',
       'TYPKODE_Factor2_30', 'TYPKODE_Factor2_31',
       'zipcode_factor_0', 'zipcode_factor_1',
       'zipcode_factor_2', 'zipcode_factor_3', 'zipcode_factor_4',
       'zipcode_factor_5', 'zipcode_factor_6', 'zipcode_factor_7',
       'zipcode_factor_8', 'zipcode_factor_9', 'zipcode_factor_10',
       'zipcode_factor_11', 'zipcode_factor_12', 'zipcode_factor_13',
       'zipcode_factor_14', 'zipcode_factor_15', 'zipcode_factor_16',
       'zipcode_factor_17', 'zipcode_factor_18', 'zipcode_factor_19',
       'BOLITYPE_aggregate_code_0',
       'BOLITYPE_aggregate_code_1', 'BOLITYPE_aggregate_code_2',
       'BOLITYPE_aggregate_code_3', 'BOLITYPE_aggregate_code_4',
       'Tariff_aggregate_code_1', 'Tariff_aggregate_code_2',
       'Tariff_aggregate_code_3', 'Tariff_aggregate_code_4',
       'Tariff_aggregate_code_5', 'Tariff_aggregate_code_6',
       'Tariff_aggregate_code_7', 'Tariff_aggregate_code_8']

#***** Create the dataset.
X=cover4_data_scaled[cols_for_testing]
y=cover4_data_scaled['ChewingYes']

#**** Cross-validation definition. 5 folds with shuffling data before segmentation
cv = KFold(n_splits=5,shuffle=True, random_state=12)

lfit_cover4 = LogisticRegression(random_state=12)

#*** Helper lists to accomodate the results from Cross-Validation repetitions
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

score_dict_cover4={}
scores_cover4=[]
scores_cover4_means={}
# Cross-Validation fold indicator
_i = 0


#*** Close any open graph
plt.close()

#**** Start the cross-validation
for train, test in cv.split(X, y):
    probas_ = lfit_cover4.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.35,label='ROC fold %d (AUC = %0.2f)' % (_i, roc_auc))
   
    #**** Accuracy
    y_pred = [1 if x >= 0.3 else 0 for x in probas_[:, 1]]
    print("Hits= ",y_pred.count(1)/len(y_pred))
    cmat=confusion_matrix(y.iloc[test], y_pred)
    specificity=cmat[0][0]/(cmat[0][0]+cmat[0][1])
    sensitivity=cmat[1][1]/(cmat[1][0]+cmat[1][1])
    f1=f1_score(y.iloc[test],y_pred)
    mcc=matthews_corrcoef(y.iloc[test],y_pred)
    score_dict_cover4={'F1_score':f1,'MCC':mcc,'Specificity':specificity,'Sensitivity':sensitivity}
    scores_cover4.append(score_dict_cover4)
    #***************************

    _i += 1

#*** Helper variable to calculate the average values. Adders
f1_s=0
mcc_s=0
specs=0
sens=0

#*** Calculate means ************
for i in range(len(scores_cover4)):
    f1_s+=dict(scores_cover4[i])['F1_score']
    mcc_s+=dict(scores_cover4[i])['MCC']
    sens+=dict(scores_cover4[i])['Sensitivity']
    specs+=dict(scores_cover4[i])['Specificity']
scores_cover4_means={'F1_score':f1_s/len(scores_cover4),'MCC':mcc_s/len(scores_cover4),'Specificity':specs/len(scores_cover4),'Sensitivity':sens/len(scores_cover4)}
print("F1= ",scores_cover4_means['F1_score']," ,MCC= ",scores_cover4_means['MCC'])
print("Specificity= ",scores_cover4_means['Specificity']," ,Sensitivity= ",scores_cover4_means['Sensitivity'])



#****** Plot the ROC framework
#**** Draw the diagonal line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
#*** Compute the AUC
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

# Plot the mean curve and fill the area between min and max values with color
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

