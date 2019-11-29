#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:53:46 2019

@author: jinlingxing
"""
import pandas as pd
import numpy as np

#from scipy.stats import t
from scipy.stats import ttest_ind
# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 
          'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 
          'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 
          'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 
          'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin',
          'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam',
          'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 
          'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri',
          'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 
          'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 
          'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 
          'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire',
          'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

def get_list_of_university_towns():
#    '''Returns a DataFrame of towns and the states they are in from the 
#    university_towns.txt list. The format of the DataFrame should be:
#    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
#    columns=["State", "RegionName"]  )
    
#    The following cleaning needs to be done:
#    1. For "State", removing characters from "[" to the end.
#    2. For "RegionName", when applicable, removing every character from " (" to the end.
#    3. Depending on how you read the data, you may need to remove newline character '\n'. '''

    #the RegionName with [edit] at the end is the name of the college town, the RegionName with (...) at the end is the university name
#    uni_towns = pd.read_csv("/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/university_towns.txt", error_bad_lines=False,header =None, names = ['RegionName'] )
    uni_towns = pd.read_csv("university_towns.txt", error_bad_lines=False,header =None, names = ['RegionName'] )
    uni_towns['RegionName'] = uni_towns['RegionName'].replace('\(.*', '', regex =True)
    #use the [edit] to verify which is the states and then remove the [edit]
    uni_towns['State_bool'] = uni_towns['RegionName'].str.contains('\[.*')
    uni_towns['RegionName'] = uni_towns['RegionName'].replace("\[.*", '', regex =True)

    for i in range(0, len(uni_towns)):
        if(uni_towns.at[i,'State_bool'] == True):
            state_name = uni_towns.at[i,'RegionName']
            uni_towns.at[i,'State'] = state_name    
        else:
            uni_towns.at[i,'State'] = state_name
        if(uni_towns.at[i,'RegionName'] == uni_towns.at[i,'State']):
            uni_towns.drop(i, inplace=True)
            
    state_towns = {state: state_abbrev for state_abbrev, state in states.items()}
    uni_towns['State'] = uni_towns['State'].map(state_towns)
    uni_towns = uni_towns[['State','RegionName']]  #column named 'state' is the abbrev, named 'State' is the full name 
#    uni_towns = uni_towns[['State','RegionName']]        
    return uni_towns


#A quarter is a specific three month period, Q1 is January through March, Q2 is April through June,
# Q3 is July through September, Q4 is October through December.
#A recession is defined as starting with two consecutive quarters of GDP decline, 
#and ending with two consecutive quarters of GDP growth.
#For this assignment, only look at GDP data from the first quarter of 2000 onward.


    
def get_recession_start():
#    '''Returns the year and quarter of the recession start time as a 
#    string value in a format such as 2005q3'''
#    gdp_level = pd.read_excel('/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/gdplev.xls')
    gdp_level = pd.read_excel('gdplev.xls') 
    gdp_level = gdp_level[7:]
    gdp_level = gdp_level.rename(columns = {'Current-Dollar and "Real" Gross Domestic Product': 'year',
                                       'Unnamed: 1': 'GDP current annual',
                                       'Unnamed: 2': 'GDP chained annual',
                                       'Unnamed: 4': 'Quarter',
                                       'Unnamed: 5': 'GDP current quarterly 2009',
                                       'Unnamed: 6': 'GDP chained quarterly 2009'})   
    gdp_level = gdp_level[['year','GDP current annual','GDP chained annual','Quarter', 'GDP current quarterly 2009','GDP chained quarterly 2009']]
    gdp_level = gdp_level.reset_index(drop=True)
    quarter_data = gdp_level[['Quarter','GDP chained quarterly 2009']]
    quarter_data = quarter_data[212:]
    quarter_data = quarter_data.reset_index(drop=True)
    recession_start_time = []

    for i in range(0, len(quarter_data)-4):
        if(quarter_data.at[i,'GDP chained quarterly 2009'] > quarter_data.at[i+1,'GDP chained quarterly 2009'] 
        and quarter_data.at[i+1,'GDP chained quarterly 2009'] > quarter_data.at[i+2,'GDP chained quarterly 2009']
        and quarter_data.at[i+2,'GDP chained quarterly 2009'] < quarter_data.at[i+3,'GDP chained quarterly 2009']
        and quarter_data.at[i+3,'GDP chained quarterly 2009'] < quarter_data.at[i+4,'GDP chained quarterly 2009']):
            recession_start_time.append(quarter_data.at[i, 'Quarter'])
#            recession_end_time.append(quarter_data.at[i+4, 'Quarter'])
#            bottom.append(quarter_data.at[i+2, 'Quarter'])
    answer2 = recession_start_time.pop()
    return answer2

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
#    gdp_level = pd.read_excel('/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/gdplev.xls') 
    gdp_level = pd.read_excel('gdplev.xls')
    gdp_level = gdp_level[7:]
    gdp_level = gdp_level.rename(columns = {'Current-Dollar and "Real" Gross Domestic Product': 'year',
                                       'Unnamed: 1': 'GDP current annual',
                                       'Unnamed: 2': 'GDP chained annual',
                                       'Unnamed: 4': 'Quarter',
                                       'Unnamed: 5': 'GDP current quarterly 2009',
                                       'Unnamed: 6': 'GDP chained quarterly 2009'})   
    gdp_level = gdp_level[['year','GDP current annual','GDP chained annual','Quarter', 'GDP current quarterly 2009','GDP chained quarterly 2009']]
    gdp_level = gdp_level.reset_index(drop=True)
    quarter_data = gdp_level[['Quarter','GDP chained quarterly 2009']]
    quarter_data = quarter_data[212:]
    quarter_data = quarter_data.reset_index(drop=True)

    recession_end_time = []

    for i in range(0, len(quarter_data)-4):
        if(quarter_data.at[i,'GDP chained quarterly 2009'] > quarter_data.at[i+1,'GDP chained quarterly 2009'] 
        and quarter_data.at[i+1,'GDP chained quarterly 2009'] > quarter_data.at[i+2,'GDP chained quarterly 2009']
        and quarter_data.at[i+2,'GDP chained quarterly 2009'] < quarter_data.at[i+3,'GDP chained quarterly 2009']
        and quarter_data.at[i+3,'GDP chained quarterly 2009'] < quarter_data.at[i+4,'GDP chained quarterly 2009']):
            recession_end_time.append(quarter_data.at[i+4, 'Quarter'])
    answer3 = recession_end_time.pop()
    return answer3

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
#    gdp_level = pd.read_excel('/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/gdplev.xls') 
    gdp_level = pd.read_excel('gdplev.xls') 
    gdp_level = gdp_level[7:]
    gdp_level = gdp_level.rename(columns = {'Current-Dollar and "Real" Gross Domestic Product': 'year',
                                       'Unnamed: 1': 'GDP current annual',
                                       'Unnamed: 2': 'GDP chained annual',
                                       'Unnamed: 4': 'Quarter',
                                       'Unnamed: 5': 'GDP current quarterly 2009',
                                       'Unnamed: 6': 'GDP chained quarterly 2009'})   
    gdp_level = gdp_level[['year','GDP current annual','GDP chained annual','Quarter', 'GDP current quarterly 2009','GDP chained quarterly 2009']]
    gdp_level = gdp_level.reset_index(drop=True)
    quarter_data = gdp_level[['Quarter','GDP chained quarterly 2009']]
    quarter_data = quarter_data[212:]
    quarter_data = quarter_data.reset_index(drop=True)

    bottom = []
    for i in range(0, len(quarter_data)-4):
        if(quarter_data.at[i,'GDP chained quarterly 2009'] > quarter_data.at[i+1,'GDP chained quarterly 2009'] 
        and quarter_data.at[i+1,'GDP chained quarterly 2009'] > quarter_data.at[i+2,'GDP chained quarterly 2009']
        and quarter_data.at[i+2,'GDP chained quarterly 2009'] < quarter_data.at[i+3,'GDP chained quarterly 2009']
        and quarter_data.at[i+3,'GDP chained quarterly 2009'] < quarter_data.at[i+4,'GDP chained quarterly 2009']):
            bottom.append(quarter_data.at[i+2, 'Quarter'])
    answer4 = bottom.pop()   
    return answer4

def convert_housing_data_to_quarters():
    
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
#    housing_data = pd.read_csv("/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/City_Zhvi_AllHomes.csv")
    housing_data = pd.read_csv("City_Zhvi_AllHomes.csv")
#    state_region = housing_data.iloc[:,1:5]     
    quarters_2000q1_2016q3 = housing_data.iloc[:,-200:]    
#    housing = pd.merge(state_region,quarters_2000q1_2016q3,how='left',left_index=True,right_index=True)    
#    housing = housing.drop(['State','RegionName'], axis=1)


#Converts the housing data to quarters and returns it as mean values in a dataframe.

# Iterate over two given columns only from the dataframe
#    housing1  = housing[:6]
    quarters_2000q1_2016q3 = quarters_2000q1_2016q3.fillna(0)
    last_mean_value = []
    for i, rows in quarters_2000q1_2016q3.iterrows():
#    print(i) # the index of housing
#    print(rows) #the columns: the housing price in each month
    #the range(0,198,3)will not calculate the last two columns, we added it here
        last_mean_value.append((rows[198]+rows[199])/2) 
        for j in range(0,198,3):
            mean_value = (rows[j]+rows[j+1]+rows[j+2])/3
            quarters_2000q1_2016q3.loc[i,j] = mean_value
            quarters_2000q1_2016q3.loc[i,j+1]= np.nan
            quarters_2000q1_2016q3.loc[i,j+2]= np.nan

#delete the first 200 columns  
    quarters_2000q1_2016q3 = quarters_2000q1_2016q3.iloc[:,200:]
    quarters_2000q1_2016q3 = quarters_2000q1_2016q3.dropna(axis=1)  
    quarters_2000q1_2016q3['last column'] = last_mean_value
    quarters_2000q1_2016q3['state'] = housing_data['State']
    quarters_2000q1_2016q3['RegionName'] = housing_data['RegionName']
#set multiple_index
#https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
#    select_index = housing_data[['State','RegionName']]
#    select_index_transposed = select_index.T
#    index_array = select_index_transposed.values.tolist()
#    index_tuples = list(zip(*index_array))
#    index = pd.MultiIndex.from_tuples(index_tuples, names=['State', 'RegionName'])
#add index in the housing and reset index
#    housing['index'] = index 
#    housing = housing.set_index('index')
    housing1 = quarters_2000q1_2016q3.set_index(["state","RegionName"])
    return housing1       


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    towns = get_list_of_university_towns()
    housing_price = convert_housing_data_to_quarters()
    housing_price.columns = range(housing_price.shape[1])  #reset index for columns
# recession_start: '2008q4'
# bottom: '2009q2'
    housing_price.rename(columns = {0:'2000q1',4:'2001q1',8:'2002q1',12:'2003q1',
                                    16:'2004q1',20:'2005q1',24:'2006q1',28:'2007q1',
                                    32:'2008q1',33:'2008q2',34: '2008q3',35:'2008q4',
                                    36: '2009q1',37: '2009q2', 40: '2010q1',44:'2011q1',
                                    48: '2012q1',52: '2013q1',56:'2014q1', 60:'2015q1',
                                    64:'2016q1',66:'2016q3'}, inplace=True)
    house = housing_price.iloc[:,35:38]
    house["ratio"] = house['2008q4'] / house['2009q2']
    house = house.drop(['2009q1'],axis=1)
    
    towns = towns.set_index(['State','RegionName'])
    towns['IsUniTown'] = 'Yes'
    towns_house = pd.merge(house, towns, how='left', left_index = True, right_index = True)  
    towns_house['IsUniTown'] = towns_house['IsUniTown'].fillna('No')
    uni_towns_house = towns_house[towns_house['IsUniTown'] == 'Yes']['ratio']
    other_towns_house = towns_house[towns_house['IsUniTown'] == 'No']['ratio']
    
#t-test   
#https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
    # calculate means
    mean1 = other_towns_house.mean() 
    mean2 = uni_towns_house.mean()
    # calculate standard errors
#    sem1 = other_towns_price['difference'].sem()     #unbiased standard error
#    sem2 = uni_towns_price['difference'].sem() 
    # standard error on the difference between the samples
#    sed = np.sqrt(sem1**2.0 + sem2**2.0)
    # calculate the t statistic
#    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
#    df = len(uni_towns_price) + len(other_towns_price) - 2
    # calculate the critical value
#    alpha = 0.01
#    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
#    p = (1 - t.cdf(abs(t_stat), df)) * 2
#this method to calculate t-test is as same as the above one    
    stat, p = ttest_ind(uni_towns_house, other_towns_house,nan_policy='omit')
    print('t=%.3f, p=%.3f' % (stat, p))
    # get different and better values
    different = False
    if p < 0.01:
        different = True
    # determine which type of town is better
    better = ""
    if mean2 > mean1:
        better = "university town"
    else:
        better = "non-university town"
    return different, p, better