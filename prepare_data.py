# It reads the csv file and creates the datasets that are necessary for running the modes.

import pandas as pd

# Read the full dataset from the csv file
disk_data=pd.read_csv("accidents_data.csv",sep=",",encoding='latin-1')

# Read the tariff file
tariff_data=pd.read_csv("occupation_tariff_codes.csv",sep=",",encoding='latin-1')

# Clean the dataset
disk_data=disk_data[~pd.isnull(disk_data.BOLITYPE)]
disk_data=disk_data[disk_data.ALDER>17]
disk_data=disk_data[~pd.isnull(disk_data.TYPKODE)].copy()

# Merge the original dataset with the tariff file. This creates the correspondence 
# between the Occupation Code and Tariff Code
result_data = pd.merge(disk_data, tariff_data, how='left',left_on='BESKTITY', right_on='Code',suffixes=('', '_y'))

# Create the codes for Housing Type
def factor_bolitype(x):
    r=-1
    if (x =='DØGN'):
        r=9
    elif(x =='KOLL'):
        r=4
    elif(x =='LAND'):
        r=3
    elif(x =='LEJL'):
        r=1
    elif(x =='RHUS'):
        r=2
    elif(x =='SOHU'):
        r=5
    elif(x =='STUE'):
        r=6
    elif(x =='UKEN'):
        r=7
    elif(x =='VILE'):
        r=8
    elif(x =='VILL'):
        r=0
    return r

# Create the codes for aggregated Housing Type
def factor_bolitype_aggregation(x):
    r=-1
    if (x >3):
        r=4
    else:
        r=x
    return r
    
# Create the codes for grouped age
def factor_alder(x):
    r=-1
    if x>17 and x<=24:
        r=1
    elif x>24 and x<=69:
        r=2
    elif x>69:
        r=3
    return r

# Create the codes for aggregated Tariff code
def factor_tariff(x):
    r=-1
    if (x >7):
        r=8
    else:
        r=x
    return r

# Create the codes for socio-economic code
def factor_typkode(x):
    r=-1
    if(x =='A1'):	
        r=2
    elif(x =='A2'):	
        r=3
    elif(x =='A3'):	
        r=4
    elif(x =='A4'):	
        r=5
    elif(x =='A5'):	
        r=6
    elif(x =='B1'):	
        r=7
    elif(x =='B2'):	
        r=8
    elif(x =='B3'):	
        r=9
    elif(x =='B4'):	
        r=10
    elif(x =='C1'):	
        r=11
    elif(x =='C2'):	
        r=12
    elif(x =='C3'):	
        r=13
    elif(x =='D1'):	
        r=14
    elif(x =='D2'):	
        r=15
    elif(x =='D3'):	
        r=16
    elif(x =='E1'):	
        r=17
    elif(x =='E2'):	
        r=18
    elif(x =='E3'):	
        r=19
    elif(x =='E4'):	
        r=20
    elif(x =='E5'):	
        r=21
    elif(x =='F1'):	
        r=22
    elif(x =='F2'):	
        r=23
    elif(x =='F3'):	
        r=24
    elif(x =='F4'):	
        r=25
    elif(x =='G1'):	
        r=26
    elif(x =='G2'):	
        r=27
    elif(x =='G3'):	
        r=28
    elif(x =='G4'):	
        r=29
    elif(x =='H1'):	
        r=30
    elif(x =='H2'):	
        r=31
    return r

# Create the codes for aggregated zip codes. In comments are the groups that 
# have been given to me. The sequence is Original zipcode -> Grouped zipcode -> Code foe Grouped zipcode
def zip_code_factor(x):
    r=-1
    if(x <= 1049):
        r= 20#'U'
    elif(x <= 2599): 
        r= 11 #"100"
    elif(x <= 2604): 
        r=9 #"101"
    elif(x <= 2609): 
        r=1 #"100"
    elif(x <= 2624): 
        r=9 #"101"
    elif(x <= 2699): 
        r=1 #"100"
    elif(x <= 2799): 
        r=9 #"101"
    elif(x <= 2859): 
        r=1 #"100"
    elif(x <= 2899): 
        r=9 #"101"
    elif(x <= 2999): 
        r=1 #"100"
    elif(x <= 3119): 
        r=0 #"090"
    elif(x <= 3139): 
        r=16 #"091"
    elif(x <= 3199): 
        r=0 #"090"
    elif(x <= 3399): 
        r=16 #"091"
    elif(x <= 3549): 
        r=0 #"090"
    elif(x <= 3659): 
        r=16 #"091"
    elif(x <= 3699): 
        r=0 #"090"
    elif(x <= 3999): 
        r=19 #"110"
    elif(x <= 4029): 
        r=5 #"071"
    elif(x <= 4039): 
        r=8 #"070"
    elif(x <= 4049): 
        r=5 #"071"
    elif(x <= 4599): 
        r=8 #"070"
    elif(x <= 4639): 
        r=10 #"080"
    elif(x <= 4670): 
        r=2 #"081"
    elif(x <= 4671): 
        r=10 #"080"
    elif(x <= 4734): 
        r=2 #"081"
    elif(x <= 4735): 
        r=10 #"080"
    elif(x <= 4770): 
        r=2 #"081"
    elif(x <= 4999): 
        r=10 #"080"
    elif(x <= 5099): 
        r=14 #"061"
    elif(x <= 5299): 
        r=14 #"061"
    elif(x <= 5319): 
        r=13 #"060"
    elif(x <= 5329): 
        r=14 #"061"
    elif(x <= 5999): 
        r=13 #"060"
    elif(x <= 6039): 
        r=7 #"051"
    elif(x <= 6090): 
        r=11 #"050"
    elif(x <= 6092): 
        r=7 #"051"
    elif(x <= 6099): 
        r=11 #"050"
    elif(x <= 6229): 
        r=7 #"051"
    elif(x <= 6239): 
        r=11 #"050"
    elif(x <= 6299): 
        r=17 #"040"
    elif(x <= 6399): 
        r=11 #"050"
    elif(x <= 6429): 
        r=7 #"051"
    elif(x <= 6499): 
        r=11 #"050"
    elif(x <= 6699): 
        r=17 #"040"
    elif(x <= 6719): 
        r=12 #"041"
    elif(x <= 6730): 
        r=17 #"040"
    elif(x <= 6739): 
        r=12 #"041"
    elif(x <= 6899): 
        r=17 #"040"
    elif(x <= 6999): 
        r=3 #"020"
    elif(x <= 7006): 
        r=7 #"051"
    elif(x <= 7129): 
        r=7 #"051"
    elif(x <= 7199): 
        r=11 #"050"
    elif(x <= 7299): 
        r=17 #"040"
    elif(x <= 7399): 
        r=11 #"050"
    elif(x <= 7999): 
        r=3 #"020"
    elif(x <= 8299): 
        r=4 #"031"
    elif(x <= 8319): 
        r=6 #"030"
    elif(x <= 8329): 
        r=4 #"031"
    elif(x <= 8339): 
        r=6 #"030"
    elif(x <= 8349): 
        r=4 #"031"
    elif(x <= 8380): 
        r=6 #"030"
    elif(x <= 8381): 
        r=4 #"031"
    elif(x <= 8461):
        r=6 #"030"
    elif(x <= 8463): 
        r=4 #"031"
    elif(x <= 8519): 
        r=6 #"030"
    elif(x <= 8529): 
        r=6 #"030"
    elif(x <= 8999): 
        r=6 #"030"
    elif(x <= 9199): 
        r=18 #"011"
    elif(x <= 9399): 
        r=15 #"010"
    elif(x <= 9429): 
        r=18 #"011"
    elif(x <= 9990): 
        r=15 #"010"
    return r

# Apply the above functions
result_data['bolitype_code'] = list(map(factor_bolitype,result_data.BOLITYPE))
result_data['business_age_group'] = list(map(factor_alder,result_data.ALDER))
result_data['Tariff_aggregate_code'] = list(map(factor_tariff,result_data.Tariff_Class))
result_data['TYPKODE_Factor2'] = list(map(factor_typkode,result_data.TYPKODE))
result_data['BOLITYPE_aggregate_code'] = list(map(factor_bolitype_aggregation,result_data.bolitype_code))
result_data['zipcode_factor'] = list(map(zip_code_factor,result_data.FTAGPOST))


#  and drop unnecessary variables
working_data=result_data.copy()
working_data.drop('Short Text in Danish',axis=1,inplace=True)
working_data.drop('Tariff_Class_y',axis=1,inplace=True)
working_data.drop('Code Nbr',axis=1,inplace=True)
working_data.drop('Code',axis=1,inplace=True)

# Create the dummy variables
global working_data_dummies
working_data_dummies=pd.get_dummies(working_data,columns=['TYPKODE_Factor2','zipcode_factor','BOLITYPE_aggregate_code','Tariff_aggregate_code'])

del(result_data)
del(disk_data)
del(tariff_data)
