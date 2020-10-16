

##############################################################################
###########################   assignment2   ##################################
##############################################################################



import pandas as pd
import numpy as np

df = pd.read_csv('olympics.csv', index_col = 0, skiprows =1)
for col in df.columns:
    if col[:2] == '01':
        df.rename(columns = {col: 'Gold' +col[4:]}, inplace = True)
    if col[:2] == '02':
        df.rename(columns = {col: 'Silver'+col[4:]}, inplace = True)
    if col[:2] == '03':
        df.rename(columns = {col: 'Bronze'+col[4:]}, inplace =True)
    if col[:1] == '№':
        df.rename(columns = {col: '#' + col[1:]}, inplace =True)

names_ids = df.index.str.split('\s\(')  # split the index by '('
df.index = names_ids.str[0]             #the country's name without abbreviation
df['ID'] = names_ids.str[1].str[:3]    #add column 'ID', the abbre of countries    

df.drop('Totals')  
df = df.drop('Totals')

#df.columns          
#df.head()


#Question 0 (Example)
#What is the first country in df?
#This function should return a Series.


# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the row for Afghanistan, which is a Series object. The assignment
    # question description will tell you the general format the autograder is expecting
    return df.iloc[0]

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 




#Question 1
#Which country has won the most gold medals in summer games?
#This function should return a single string value.
def answer_one():
    #return "YOUR ANSWER HERE"
    df['Gold'].argmax()
    return df['Gold'].argmax()
#dff = pd.Index([17, 69, 33, 5, 0, 74, 0])
#dff.max()
#dff.argmax()




#Question 2
#Which country had the biggest difference between their summer and winter gold medal counts?
#This function should return a single string value.
def answer_two():
    #return "YOUR ANSWER HERE"
    difference = df['Gold'] - df['Gold.1']
    return difference.argmax()



#Question 3
#Which country has the biggest difference between their summer gold medal counts
# and winter gold medal counts relative to their total gold medal count?
#Summer Gold−Winter GoldTotal Gold
#Only include countries that have won at least 1 gold in both summer and winter.
#This function should return a single string value.
    
    ##if((df['Gold']>1) & (df['Gold.1']>1)):  true/false results
def answer_three():
    
    new_df = df[(df['Gold']>=1) & (df['Gold.1']>=1)]  #new data frame 
    ddd = (new_df['Gold'] - new_df['Gold.1']).abs()/(new_df['Gold.2'])
    ddd.argmax()
    answer3 = ddd.argmax()
    return answer3


#Question 4
#Write a function to update the dataframe to include a new column called "Points" 
#which is a weighted value where each gold medal counts for 3 points, silver medals
# for 2 points, and bronze mdeals for 1 point. The function should return only the
# column (a Series object) which you created.
#This function should return a Series named Points of length 146
def answer_four():
    weight = df['Gold.2']*3 + df['Silver.2']*2+ df['Bronze.2']*1
    df['Points'] = weight
    answer4 = df['Points']
    return answer4




#Part 2
#For the next set of questions, we will be using census data from the United States Census Bureau.
# Counties are political and geographic subdivisions of states in the United States.
# This dataset contains population data for counties and states in the US from 2010 to 2015. 
#See this document for a description of the variable names.
#The census dataset (census.csv) should be loaded as census_df. Answer questions using this as appropriate.


census_df = pd.read_csv('census.csv')
census_df.head()

#Question 5
#Which state has the most counties in it? (hint: consider the sumlevel key carefully!
# You'll need this for future questions too...)
#This function should return a single string value.
#040: state
#050: county
def answer_five():
    
    new_census_df = census_df[census_df['SUMLEV']==50]# filter The "counties" with 'SUMLEV' == 40
    new_group = new_census_df.groupby('STNAME').count() #.count() function counts the number of values in each column.
    answer5 = new_group['COUNTY'].idxmax()
    return answer5

#Question 6
#Only looking at the three most populous counties for each state,
# what are the three most populous states (in order of highest population to lowest population)? Use CENSUS2010POP.
#This function should return a list of string values.
def answer_six():   
    new_census_df = census_df[census_df['SUMLEV']==50]# filter The "counties" with 'SUMLEV' == 40
    cen2010 = new_census_df[['STNAME','CTYNAME' , 'CENSUS2010POP']]
    popular_counties_df = cen2010.groupby('STNAME')['CENSUS2010POP'].apply(lambda x: x.nlargest(3).sum()).nlargest(
    3)
    answer6 = popular_counties_df.index.tolist()  
    return answer6

#Question 7
#Which county has had the largest absolute change in population within the period 2010-2015? 
#(Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, 
#you need to consider all six columns.)
#e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130, 
#then its largest change in the period would be |130-80| = 50.
#This function should return a single string value.

########   tried the only one row first  ########

#row0 = largest_change_df.iloc[0]        
#for i in range(0,6):
#    min_value = row0[0]
#    max_difference = 0
#    if(max_difference < row0[i] - min_value):
#        max_difference = row0[i]- min_value
#    elif(min_value > row0[i]):
#        min_value = row0[i]
def answer_seven():
    new_census_df = census_df[census_df['SUMLEV']==50]# filter The "counties" with 'SUMLEV' == 40
    cen2010_2015 = new_census_df[['STNAME', 'CTYNAME','POPESTIMATE2010', 'POPESTIMATE2011',
                              'POPESTIMATE2012', 'POPESTIMATE2013', 'POPESTIMATE2014', 'POPESTIMATE2015']]


    largest_change_df = cen2010_2015.groupby('CTYNAME').sum()
    max_difference = 0
    for i, row in largest_change_df.iterrows():
    #print(i)  #the index: the name of counties
    #print(row) #the columns: the population in each county 
        for j in range(0,6): 
            min_value = row[0]  #set the first value of each row as the minimum value and it will update 
            if(max_difference < row[j] - min_value):
                max_difference = row[j] - min_value
                index_county = i
            elif(min_value > row[j]):
                min_value =row[j]
    index_county

    return index_county


#Question 8
#In this datafile, the United States is broken up into four regions using the "REGION" column.
#Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington',
# and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
#This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME'] 
#and the same index ID as the census_df (sorted ascending by index).
def answer_eight():
    new_census_df = census_df[census_df['SUMLEV']==50]# filter The "counties" with 'SUMLEV' == 40
    cen2014_2015 = new_census_df[['STNAME', 'REGION','CTYNAME'  ,  'POPESTIMATE2014', 'POPESTIMATE2015']]
    cen2014_2015_region = cen2014_2015[cen2014_2015['REGION'] < 3 ] #regions 1 or 2
    substring = 'Washington'# substring to be searched 
    cen2014_2015_region['substr_name'] = cen2014_2015_region['CTYNAME'].str.find(substring) 
#If the substring doesn’t exist in the text, -1 is returned.
# creating and passsing series to new column 
    wash_name =  cen2014_2015_region[cen2014_2015_region['substr_name'] == 0]
    wash_name_2014_2015 = wash_name[wash_name['POPESTIMATE2015'] > wash_name['POPESTIMATE2014'] ]
    return_wash_name = wash_name_2014_2015[['STNAME', 'CTYNAME']]

    return return_wash_name 


