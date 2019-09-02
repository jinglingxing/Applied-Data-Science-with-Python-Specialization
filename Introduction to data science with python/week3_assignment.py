#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:04:06 2019

@author: jinlingxing
"""
import pandas as pd
import numpy as np
####################################     Question 1    #################################### 
energy = pd.read_excel("/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/EnergyIndicators.xls", usecols=[2,3,4,5])
energy = energy.rename(columns = {'Unnamed: 2':'Country', 'Unnamed: 3':'Energy Supply', 'Unnamed: 4':'Energy Supply per Capita','Unnamed: 5': '% Renewable' })
energy = energy[17:244]
energy['Energy Supply'] = energy['Energy Supply']* 1000000
#set "........" to be 'NaN ' 

energy['Energy Supply'] =  energy['Energy Supply'].replace('\.+', np.NAN, regex =True)
energy['Energy Supply per Capita'] =  energy['Energy Supply per Capita'].replace('\.+', np.NAN, regex =True)

#only special symbols need \, letters don't need that.
#s =  pd.DataFrame({'A': [0, 1, 2, 3, 4],
#                   'B': [5, 6, 7, 8, 9],
#                   'C': ['aaaa++++ssdsadsaaa', 'baaaasdsasaafr', 'cfskmsdkasaaa', 'ddsdkkkkaa', 'edsksmsdk']})
#s['C'] = s['C'].replace('\++', 'A', regex =True)
#s['C'] = s['C'].replace('a+', 'W', regex =True)
energy = energy.replace({"Republic of Korea": "South Korea",
                "United States of America": "United States",
                "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                "China, Hong Kong Special Administrative Region": "Hong Kong" })

#'Switzerland17' should be 'Switzerland'
energy['Country'] = energy['Country'].replace('\d', '', regex =True )
#in regular expression, .* will be greedy and try to match as much as possible
energy['Country'] = energy['Country'].replace('\(.*\)', '', regex =True )

gdp = pd.read_csv("/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/world_bank.csv")
gdp = gdp[3:]
gdp = gdp.replace({"Korea, Rep.": "South Kons =rea", 
                     "Iran, Islamic Rep.": "Iran",
                     "Hong Kong SAR, China": "Hong Kong"} )

#Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15).
gdp = gdp[1:] #remove the real columns name
gdp =gdp.rename(columns ={ 'Data Source' : 'Country' , 
                          'Unnamed: 50': '2006',
                          'Unnamed: 51': '2007',
                          'Unnamed: 52': '2008',
                          'Unnamed: 53': '2009',
                          'Unnamed: 54': '2010',
                          'Unnamed: 55': '2011', 
                          'Unnamed: 56': '2012',
                          'Unnamed: 57': '2013',
                          'Unnamed: 58': '2014',
                          'Unnamed: 59': '2015'})
gdp = gdp[['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]

ScimEn = pd.read_excel("/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/scimagojr_3.xlsx")
ScimEn1 = ScimEn 
ScimEn = ScimEn[0:15]  #rank 1-15
ScimEn_15 = ScimEn[0:15]  #rank 1-15

ScimEn_energy = pd.merge(ScimEn, energy , how = 'left', left_on = 'Country', right_on = 'Country')
ScimEn_energy_gdp = pd.merge(ScimEn_energy, gdp, how ='left', left_on = 'Country', right_on = 'Country')
ScimEn_energy_gdp = ScimEn_energy_gdp.set_index('Country')

####################################     Question 2    #################################### 
outer_ScimEn1_energy = pd.merge(ScimEn1, energy, how ='outer', left_on = 'Country', right_on = 'Country')
outer_ScimEn1_energy_gdp = pd.merge(outer_ScimEn1_energy, gdp, how ='outer', left_on = 'Country', right_on = 'Country')

inner_ScimEn_energy = pd.merge(ScimEn, energy, how ='inner', left_on = 'Country', right_on = 'Country')
inner_ScimEn_energy_gdp = pd.merge(inner_ScimEn_energy , gdp, how ='inner', left_on = 'Country', right_on = 'Country')

result = len(outer_ScimEn1_energy_gdp) - len(inner_ScimEn_energy_gdp)

####################################     Question 3    #################################### 
#What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
#This function should return a Series named avgGDP with 15 countries and their average GDP sorted in descending order.

#axis = 1 means row, axis = 0 means column
AVG = ScimEn_energy_gdp.iloc[:, 10:].mean(axis = 1, skipna = True)
avgGDP = AVG.sort_values(ascending = False) 
    
####################################     Question 4     #################################### 
#By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
#This function should return a single number.

avgGDP.index[5]
df_4 = ScimEn_energy_gdp.reset_index()
df_4_avgGDP = df_4[df_4['Country'] == avgGDP.index[5]]
result_4 = (df_4_avgGDP['2015']-df_4_avgGDP['2006']).values


####################################     Question 5     #################################### 
# What is the mean Energy Supply per Capita?
# This function should return a single number.
energy['Energy Supply per Capita'].dropna().mean()
#energy['Energy Supply per Capita'].mean(axis=0, skipna = False)


####################################     Question 6     #################################### 
#What country has the maximum % Renewable and what is the percentage?
#This function should return a tuple with the name of the country and the percentage.

#idxmax() only for numeric values

max_energy = energy['% Renewable'].max()
country_max_energy = energy[energy['% Renewable'] == max_energy]
country_max_energy =country_max_energy.set_index('Country')
country_max_energy['% Renewable']

####################################     Question 7     #################################### 
#Create a new column that is the ratio of Self-Citations to Total Citations. 
#What is the maximum value for this new column, and what country has the highest ratio?
#This function should return a tuple with the name of the country and the ratio.

ratio = ScimEn['Self-citations']/ ScimEn['Citations']
ScimEn['ratio'] = ratio
max_ratio = ScimEn['ratio'].max()
country_max_ratio = ScimEn[ScimEn['ratio'] == max_ratio]
country_max_ratio =country_max_ratio.set_index('Country')
country_max_ratio['ratio']


####################################     Question 8     #################################### 
#Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
#What is the third most populous country according to this estimate?
#This function should return a single string value.
population = energy['Energy Supply']/ energy['Energy Supply per Capita']
energy['population'] = population
sort_energy = energy.sort_values(by = ['population'], ascending = False)
third_popu = sort_energy.iloc[2]
third_popu['Country']


####################################     Question 9     #################################### 
#Create a column that estimates the number of citable documents per person. 
#What is the correlation between the number of citable documents per capita and the energy supply per capita? 
#Use the .corr() method, (Pearson's correlation).
#This function should return a single number.
#(Optional: Use the built-in function plot9() to visualize the relationship between Energy Supply per Capita vs.
# Citable docs per Capita)
df_9 = ScimEn_energy[['Citations per document','Energy Supply per Capita']]

df_9['Citations per document'].corr(df_9['Energy Supply per Capita'])
#df['A'].corr(df['B']) return a value
#correlation = df_9.corr(method = 'pearson') return a data frame



####################################     Question 10     #################################### 
#Create a new column with a 1 if the country's % Renewable value is at or 
#above the median for all countries in the top 15, 
#and a 0 if the country's % Renewable value is below the median.


mean_renew = energy['% Renewable'].mean()
energy['comparision']= None
for i in range(len(energy)):
    if(energy.iloc[i]['% Renewable'] >= mean_renew):
        #energy.iloc[i].at['comparision'] = 0, iloc only works on the integer values 
        energy.at[i,'comparision'] = 1
    else:
        energy.at[i,'comparision'] = 0

####################################     Question 11     ####################################    
#Use the following dictionary to group the Countries by Continent, 
#then create a dateframe that displays the sample size (the number of countries
#in each continent bin), and the sum, mean, and std deviation for the estimated
#population of each country.
ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
df_cont = pd.DataFrame.from_dict(ContinentDict,orient='index',columns =['continent'])
df_cont_1 = df_cont.reset_index()
df_cont_2 = df_cont_1.rename(columns ={'index' : 'Country' })
df_energy_cont = pd.merge(df_cont_2,energy, how ='left', left_on = 'Country', right_on = 'Country')
df_11 = df_energy_cont[['population','continent']]
df_11 = df_11.groupby('continent')['population'].agg(['size','sum','mean','std'])

####################################     Question 12     ####################################   
#Cut % Renewable into 5 bins. Group Top15 by the Continent, 
#as well as these new % Renewable bins. How many countries are in each of these groups?
#This function should return a Series with a MultiIndex of Continent, 
#then the bins for % Renewable. Do not include groups with no countries.
df_12 = df_energy_cont[['% Renewable','continent']]
df_12 = df_12.groupby('continent')['% Renewable'].size()



