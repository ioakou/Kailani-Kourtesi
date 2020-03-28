# It creates the graph of the ordered segmentation of 
# Average probability vs the Actual sales for Chewing Damage.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from prepare_data import working_data_dummies
# import prepare_data as prep_dt


#***** There is a restriction on the age. It must be less than 70
# global working_data_dummies
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

# Create the train and test datasets
X=cover4_data_scaled[cols_for_testing]
Y=cover4_data_scaled['ChewingYes']
X_train_cover4,X_test_cover4,Y_train_cover4,Y_test_cover4=train_test_split(X,Y,test_size=0.2)

lfit_cover4 = LogisticRegression(random_state=12)
probas_ = lfit_cover4.fit(X_train_cover4, Y_train_cover4).predict_proba(X_test_cover4)

# Create a dictionary with Actuals and Predicted values.
# It is my standard procedure to create a data frame afterwards
d={'Actuals':Y_test_cover4,'Predicted':probas_[:,1]}

# Create the data frame
act_prob=pd.DataFrame(d)

# Sort the Predicted probabilities
act_prob.sort_values('Predicted',ascending=False,inplace=True)

# Create a step for segmenting the Predicted Probabilities
max_j=np.int(np.floor(len(act_prob)/20))

x_label=np.arange(20)

# Utility variables for storing purposes
c4_min=[]
c4_max=[]
c4_avg=[]
c4_hit=[]

# Loop in all 20 segments
for j in range(20):
    start = max_j*j 
    end  = (max_j*(j+1))-1
    if(j==20):
        end=len(act_prob)
    lst = act_prob[start:end]
    c4_min.append(min(lst['Predicted']))
    c4_max.append(max(lst['Predicted']))
    c4_avg.append(np.mean(lst['Predicted']))
    # Calculate the Hit Rate
    c4_hit.append(sum(lst['Actuals'])/(end-start+1))

d={'Min':c4_min,'Max':c4_max,'Avg. Predicted Sales':c4_avg,'Actual Hit rate':c4_hit}
c4_full_df=DataFrame(d)
plt.xticks(x_label)
plt.plot(x_label,c4_full_df['Avg. Predicted Sales'])
plt.plot(c4_full_df['Actual Hit rate'])

plt.legend(loc="upper right")