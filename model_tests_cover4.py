from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import numpy as np
from sklearn.utils import class_weight
from statsmodels.graphics.factorplots import interaction_plot
from sklearn.model_selection import ShuffleSplit
from prepare_data import working_data_dummies

cover4_data=working_data_dummies[working_data_dummies.ALDER<70]
cover4_data_scaled=cover4_data.copy()
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
response_variable='ChewingYes'


X=cover4_data_scaled[cols_for_testing]
Y=cover4_data_scaled[response_variable]

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,shuffle=True)
cv_def = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

#**** Logistic regression
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()#(random_state=12)
lfit_cover4=logistic.fit(X_train,Y_train)
probas_=lfit_cover4.predict_proba(X_test)
y_pred = [1 if x >= 0.3 else 0 for x in probas_[:, 1]]
scores_glm_cover4=np.mean(cross_val_score(lfit_cover4,X_test,Y_test,cv=5))


#Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bNB=BernoulliNB()
bNBfit_cover4=bNB.fit(X_train,Y_train)
scores_bNB_cover4=np.mean(cross_val_score(bNBfit_cover4,X_train,Y_train,cv=5))#,scoring='roc_auc'))

#Tree classifier
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
treefit_cover4=tree.fit(X_train,Y_train)
scores_Tree_cover4=np.mean(cross_val_score(treefit_cover4,X_train,Y_train,cv=cv_def,scoring='f1'))#,scoring='roc_auc'))
preds_tree_cover4=treefit_cover4.predict(X_train)

#*** Random Forests
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=5,random_state=42)
rf_fit_cover4=rf.fit(X_train,Y_train)
scores_rf_cover4=np.mean(cross_val_score(rf_fit_cover4,X_train,Y_train,cv=5))#,scoring='roc_auc'))

#**** Neural Networks
#MLP
import pandas as pd
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(max_iter=500, alpha=.0001,random_state=42)
mlpfit_cover4=mlp.fit(X_train,Y_train)
scores_NN_cover2=np.mean(cross_val_score(mlpfit_cover4,X_train,Y_train,cv=5,scoring='roc_auc'))

#RNN
# create model
model = Sequential()
model.add(Dense(20, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
estimator=model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, Y_train, validation_split=0.20, batch_size=400, epochs=20, verbose=0)
score = model.evaluate(X_train, Y_train, verbose=0)
score_accuracy = score[1]