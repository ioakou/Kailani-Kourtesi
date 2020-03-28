
#**** This is graph examples. It creates the variable with all the necessary values (ss) and you just copy it from Variable explorer
#**** in Excel
from prepare_data import working_data, working_data_dummies

ss=working_data.BOLITYPE_aggregate_code
max(ss)
ss=working_data.BOLITYPE
max(ss)
ss=working_data.TYPKODE_Factor2
max(ss)
ss=working_data.ALDER
max(ss)
ss=working_data.business_age_group
max(ss)
#*** For graph 1
chew_distr=len(working_data.ChewingYes[working_data.ChewingYes==1])/len(working_data)
len(working_data.CriticalIllnessYes[working_data.CriticalIllnessYes==1])/len(working_data)
len(working_data.DiseaseYes[working_data.DiseaseYes==1])/len(working_data)
