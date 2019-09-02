#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:23:55 2019

@author: jinlingxing
"""
#########################################################################################
                                  #Merging Dataframes#
#########################################################################################
import pandas as pd

df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df

#add new column with different values for each row
df['Date']= ['Dec', 'May', 'Jun']
df

df['Delivered'] = True
df
#the problems above is we can only add a few items
df['Fedback'] = ['Good', None, 'Bad']
df
adf = df.reset_index()
adf
adf['Date']= pd.Series({0: 'Aug', 2: 'Jan'})
adf  #it added NaN for index 1, using this approach, we can ignore the data we don't know

#join two dataframe together

staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')
print(staff_df.head())
print()
print(student_df.head())


pd.merge(staff_df, student_df, how ='outer', left_index = True, right_index = True)
pd.merge(staff_df, student_df, how ='inner', left_index = True, right_index = True)

#these examples we called set addition
#get all the members (staffs or students) that they are staffs
pd.merge(staff_df, student_df, how ='left' , left_index =True, right_index = True) 
#get all the members (staffs or students) that they are students
pd.merge(staff_df, student_df, how ='right' , left_index =True, right_index = True) 

#don't use indices, use the column to merge
#need to run it seperately, don't run the pd.merge above
staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
aaa =pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')


#if the data frame has conflicts, add location , fot the students, the location is their home address
#the location_x is the left join, the location_y is the right join
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])
bbb = pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')



#products = products.reset_index('ProductID')
#answer = pd.merge(products, invoices, how = 'left', left_on = 'ProductID', right_on = 'ProductID')
#answer = pd.merge(products, invoices, how = 'left', left_index = True, right_on = 'ProductID')


#multi-indexing and multiple columns
staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
staff_df
student_df
pd.merge(staff_df, student_df, how='inner', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])




#########################################################################################
                     #Idiomatic Pandas: Making Code Pandorable#
#########################################################################################
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/census.csv')
df

(df.where(df['SUMLEV']==50)
    .dropna()
    .set_index(['STNAME', 'CTYNAME'])
    .rename(columns = {'ESTIMATESBASE2010' : 'Estimates base 2010'}))

#the above one is more pandarable
#df = df[df['SUMLEV']==50]
#df.set_index(['STNAME','CTYNAME'], inplace=True)
#df.rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'})

#Suppose we are working on a DataFrame that holds information 
#on our equipment for an upcoming backpacking trip.

#Can you use method chaining to modify the DataFrame df in one statement 
#to drop any entries where 'Quantity' is 0 and rename the column 'Weight' to 'Weight (oz.)'?
df =(df.where(df['Quantity']==1)
      .rename(columns = {'Weight':'Weight (oz.)' }))
#这个上面的返回的是Nan
#下面的才是真的drop了
print(df.drop(df[df['Quantity'] == 0].index).rename(columns={'Weight': 'Weight (oz.)'}))

#the beginner

# If you're doing this as part of data cleaning your likely to find yourself 
#wanting to add new data to the existing DataFrame.

def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})

aa = df.apply(min_max, axis=1)

#But this parameter is really the parameter of the index to use.
# So, to apply across all rows, you pass axis equal to one

#In that case you just take the row values and add in new columns indicating the max and minimum scores. 


def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    row['max']= np.max(data)
    row['min']= np.min(data)

df.apply(min_max, axis =1)
df.head()

#But to get the most of the discussions you'll see online,
#you're going to need to know how at least read lambdas.
#Here's a one line example of how you might calculate the max of the columns using the apply function.

rows = ['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']
df.apply(lambda x: np.max(x[rows]), axis=1)

#########################################################################################
                                      #GroupBy#
#########################################################################################
df = df[df['SUMLEV']==50]
##timeit -n 10
for state in df['STNAME'].unique():
    avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])
    print('Counties in state ' + state + ' have an average population of ' + str(avg))


#the groupby one is faster

for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state ' + group + ' have an average population of ' + str(avg))

df.head()

df = df.set_index('STNAME')

#spit tasks
def fun(item):
    if item[0]<'M':
        return 0
    if item[0]<'Q':
        return 1
    return 2

#divide df into 3 parts
for i in df.groupby(fun):
    print (str(i))
    
for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')


#suppose we are interested in finding our total weight for each category.
# Use groupby to group the dataframe, and apply a function to calculate the 
#total weight (Weight x Quantity) by category.

print(df.groupby('Category').apply(lambda df,a,b: sum(df[a] * df[b]), 'Weight (oz.)', 'Quantity'))


# Or alternatively without using a lambda:
# def totalweight(df, w, q):
#        return sum(df[w] * df[q])
#        
# print(df.groupby('Category').apply(totalweight, 'Weight (oz.)', 'Quantity'))


df.groupby('STNAME').agg({'CENSUS2010POP': np.average})
print(type(df.groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']))
print(type(df.groupby(level=0)['POPESTIMATE2010']))

(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
    .agg({'avg': np.average, 'sum': np.sum}))


(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'avg': np.average, 'sum': np.sum}))

(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum}))




#########################################################################################
                                      #scale#
#########################################################################################

df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df
df['Grades'].astype('category').head()

grades = df['Grades'].astype('category',
                             categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered=True)
grades.head()


#feature extraction(third course), boolean variables ==> dummy variables.
s = pd.Series(['Low', 'Low', 'High', 'Medium', 'Low', 'High', 'Low'])
print(s.astype('category', categories =['Low','Medium','High'], ordered =True))


df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})
pd.cut(df['avg'],10)





#########################################################################################
                                      #pivot#
#########################################################################################
#A pivot table is itself a data frame, where the rows represent one variable
#that you're interested in, the columns another, and the cell's some aggregate value. 
df = pd.read_csv('/Users/jinlingxing/dataScienceStudy/Introduction_data_science_by_python/cars.csv')
df.head()
df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)
#functions can be two, the margins are the sum or min of each columns and rows
df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True)

#print(Bikes)
#import numpy as np
# Your code here
#pd.pivot_table(Bikes,index = ['Manufacturer', 'Bike Type'], aggfunc = np.mean)

#   Bike Type Manufacturer  Price  Rating
#0   Mountain            A    400       8
#1   Mountain            A    600       9
#2       Road            A    400       4
#3       Road            A    450       4
#4   Mountain            B    300       6
#5   Mountain            B    250       5
#6       Road            B    400       4
#7       Road            B    500       6
#8   Mountain            C    400       5
#9   Mountain            C    500       6
#10      Road            C    800       9
#11      Road            C    950      10
#                        Price  Rating
#Manufacturer Bike Type               
#A            Mountain   500.0     8.5
#             Road       425.0     4.0
#B            Mountain   275.0     5.5
#             Road       450.0     5.0
#C            Mountain   450.0     5.5
#             Road       875.0     9.5




















