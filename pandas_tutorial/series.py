#starting with pandas series
#creating a simple pandas series
import pandas as pd
a=pd.Series([23,-5,-8,56,47,-89,12])
print(a)
print(type(a))
#printing index
a1=a.index
print(a1)
print(type(a1))
#printing values
a2=a.values
print(a2)
print(type(a2))
#indexing
a=pd.Series([23,-5,-8,56,47,-89,12],index=['a','b','c','d','e','f','j'])
print(a)
print(type(a.index))
print(type(a.values))

##selecting internal element
print(a[3])
print(a[6])
#spliting

print(a[0:])
print(a[::-1])
print(a[1:4])

#Assigning Values
a[1]=15
print(a)

a['d']=65
print(a)

#numpy
import numpy as np
arr=np.array([11,21,31,41,51])
b1=pd.Series(arr)
print(b1)

#filtering values
f=b1[b1>=30]
print(f)

#Operating
print(b1/2)

#applying function
def my_function(x):
    z=x+33
    return z

print(b1.apply(my_function))

#Evaluating Values
a=pd.Series([1,0,1,3,3,3,3,3,4,4,5], index=['a','b','c','d','e','f','i','j','k','l','m'])
print(a)
print(a.unique())
bb=a.value_counts()
print(bb)

#NaN Values
import numpy as np
s2=pd.Series([5,-3,np.NaN,14])
print(s2)
print(s2.isnull())
print(s2.notnull())

print(s2[s2.notnull()])
print(s2[s2.notnull()])

#series as dictionary
c={'red':200,'blue':1000,'yellow':500,'orange':100}
series=pd.Series(c)
print(series)

#operations between series
d={'red':400,'yellow':1000,'black':700}
series2=pd.Series(d)
print(series2)

print(series+series2)

#https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Series.html
#Fremont_weather dataset
import numpy as np
import pandas as pd
df=pd.DataFrame(
	[['Jan',58,42,74,22,2.95],
	['Feb',61,45,78,26,3.02],
	['Mar',65,48,84,25,2.34],
	['Apr',67,50,92,28,1.02],
	['May',71,53,98,35,0.48],
	['Jun',75,56,107,41,0.11],
	['Jul',77,58,105,44,0.0],
	['Aug',77,59,102,43,0.03],
	['Sep',77,57,103,40,0.17],
	['Oct',73,54,96,34,0.81],
	['Nov',64,48,84,30,1.7],
	['Dec',58,42,73,21,2.56]],
	#index = [0,1,2,3,4,5,6,7,8,9,10,11],
	columns = ['month','av_high','avg_low','record_high','record_low','avg_pr'])
#print(df)
#print(df.head())
#print(df.tail())
#print(df.describe())

##reading file
df=pd.read_csv('Fremont_weather.txt')
#print(df)
#print(df.columns)
#print(df.index)
print(df['record_high'].max())
print(df['record_high'].min())
print(df['record_high'].sum())
print()


data={'color':['blue','green','yellow','red','white'],
      'object':['ball','pen','pencil','paper','mug'],
      'price':[1.2,1.0,0.6,0.9,1.7]}
#print(data)
frame=pd.DataFrame(data)
#print(frame)

frame2=pd.DataFrame(data, columns=['object','price'])
#print(frame2)

frame2=pd.DataFrame(data,index=['one','two','three','four','five'])
#print(frame2)

#print(frame2.columns)
#print(frame2.index)

#print(frame2.values)

#slicing
#print(frame2.ix[2])
#print(frame2.ix[[2,4]])

#single value
print(frame2['object'][3])

print(frame2.head())
print(frame2.tail())
####
###DataFrame using list
df=pd.DataFrame(
	[['Jan',58,42,74,22,2.95],
	['Feb',61,45,78,26,3.02],
	['Mar',65,48,84,25,2.34],
	['Apr',67,50,92,28,1.02],
	['May',71,53,98,35,0.48],
	['Jun',75,56,107,41,0.11],
	['Jul',77,58,105,44,0.0],
	['Aug',77,59,102,43,0.03],
	['Sep',77,57,103,40,0.17],
	['Oct',73,54,96,34,0.81],
	['Nov',64,48,84,30,1.7],
	['Dec',58,42,73,21,2.56]],
	#index = [0,1,2,3,4,5,6,7,8,9,10,11],
	columns = ['month','av_high','avg_low','record_high','record_low','avg_pr'])
print(df)

del df['month']
print(df)


df['new']=12
print(df)

df['ert']=[11,21,31,42,51,61,71,81,91,101,111,121]
print(df)

############
import pandas as pd
a=pd.read_csv('F:/freelencing projects/pandas_tutorial/nyc_weather.csv')
#print(a)
#what is the maximum temperature

print(a['Temperature'].max())
#date on which it rains
print(a['EST'][a['Events']=='Rain'])

a.fillna(0,inplace=True)
print(a['WindSpeedMPH'].mean())

#############
import pandas as pd
x=pd.read_csv('F:/freelencing projects/pandas_tutorial/weather_data.csv')
'''#print(x)
#print(x.shape)
rows,columns=x.shape
print(rows)
print(columns)
#print(x.head(2))
print(x.tail(3))
print(x.day)
print(type(x.day))
print(x[['temperature','day']][x['event']=='Snow'])
print(x[x.temperature>=32])
print(x[x.temperature==x.temperature.max()])
print(x[['day','windspeed']][x.temperature==x.temperature.max()])
print(x.set_index('day'))
print(x)

print(x.set_index('day',inplace=True))
#print(x)

print(x.loc['1/5/2017'])'''
print(x)

print(x.set_index('event',inplace=True))
print(x.loc['Snow'])
#######

import pandas as pd
a=pd.read_csv('F:/freelencing projects/pandas_tutorial/stock_data.csv')
print(a)


import pandas as pd
a=pd.read_csv('F:/freelencing projects/pandas_tutorial/stock_data.csv',header=None, names=["ticket","eps","revenue","price","people"] )
print(a)


import pandas as pd
a=pd.read_csv('F:/freelencing projects/pandas_tutorial/stock_data.csv',nrows=2)
print(a)


import pandas as pd
a=pd.read_csv('F:/freelencing projects/pandas_tutorial/stock_data.csv')
print(a)


import pandas as pd
a=pd.read_csv('F:/freelencing projects/pandas_tutorial/stock_data.csv', na_values={
        'eps':["not available","n.a."],
        'revenue':[-1],
        'people':["not available","na"]
        })
print(a)

a.to_csv("new.csv")
a.to_excel("new.xlsx")

a.to_csv("new.csv",index=False)

df.to_csv("new.csv",header=False, columns=['tickets','eps'])

###

import pandas as pd
a=pd.read_csv('F:/freelencing projects/pandas_tutorial/stock_data.csv')
print(a)

def convert_people_cell(cell):
    if cell=="n.a.":
        return 'John'
    return cell


import pandas as pd
a=pd.read_csv('F:/freelencing projects/pandas_tutorial/stock_data.csv',converters={'people':convert_people_cell})
print(a)

#####
import pandas as pd
c=pd.read_csv('F:/freelencing projects/pandas_tutorial/fillna.csv')
print(c)
print(type(c.day[0]))

import pandas as pd
c=pd.read_csv('F:/freelencing projects/pandas_tutorial/fillna.csv',parse_dates=["day"])
#print(c)
#print(type(c.day[0]))
c.set_index('day',inplace=True)
print(c)
new_c=c.fillna(0)
print(new_c)
new_c=c.fillna(method="ffill")
print(new_c)

d=c.interpolate()
print(d)


d=c.interpolate(method="time")
print(d)

new_c1=c.dropna()
print(new_c1)

new_c1=c.dropna(how="all")
print(new_c1)


new_c1=c.dropna(thresh=2)
print(new_c1)

print(c)
dt=pd.data_range("01-01-2017","01-11-2017")
idx=pd.DatatimeIndex(dt)
c=c.reindex(idx)
print(c)








new_c=c.fillna(method="ffill",limit=1)
print(new_c)


new_c=c.fillna(method="bfill")
print(new_c)


new_c=c.fillna(method="bfill",axis="columns")
print(new_c)





c=pd.read_csv('F:/freelencing projects/pandas_tutorial/fillna.csv',parse_dates=["day"])
new_c=c.fillna({
        'temperature':0,
        'windspeed':0,
        'event':'no event'})
print(new_c)

#####
import pandas as pd
s1=pd.read_csv('F:/freelencing projects/pandas_tutorial/replace.csv')
print(s1)

new_s1=s1.replace(-99999,np.NaN)
print(new_s1)


new_s1=s1.replace([-99999,-888888],np.NaN)
print(new_s1)


new_s1=s1.replace({
        'temperature':-99999,
        'windspeed':-99999,
        'event':0
        },np.NaN)
print(new_s1)


print(s1)
new_s1=s1.replace({
        -99999:np.NaN,
        '0':'sunny'})
print(new_s1)




####
import pandas as pd
w=pd.read_csv('F:/freelencing projects/pandas_tutorial/groupby.csv')
print(w)
l=w.groupby('city')
'''for city, city_w in l:
    print(city)
    print(city_w)'''
    
l.get_group('mumbai')
l.max()
l.min()
l.describe()

import pandas as pd


ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

#print (df)

#print (df.groupby('Team').groups)


grouped = df.groupby('Year')
#for name,group in grouped:
   #print (name)
   #print (group)
print( grouped.get_group(2014))



import pandas as pd

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

grouped = df.groupby('Year')
print (grouped.get_group(2014))



grouped = df.groupby('Year')
print (grouped['Points'].agg(np.mean))
print (grouped.agg(np.size))

print( grouped['Points'].agg([np.sum, np.mean, np.std]))

print (df.groupby('Team').filter(lambda x: len(x) >= 3))

########
import pandas as pd
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                      index=[0, 1, 2, 3])
print(df1)


df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                       index=[4, 5, 6, 7])


df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                       'B': ['B8', 'B9', 'B10', 'B11'],
                       'C': ['C8', 'C9', 'C10', 'C11'],
                       'D': ['D8', 'D9', 'D10', 'D11']},
                      index=[8, 9, 10, 11])


df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                      'D': ['D2', 'D3', 'D6', 'D7'],
                      'F': ['F2', 'F3', 'F6', 'F7']},
                     index=[2, 3, 6, 7])
print(df4)

r1=[df1,df4]
#print(pd.concat(r1,join='inner'))
print(pd.concat(r1))
print(result)

result1=pd.concat(r1,join='inner',axis=1)
print(result1)

result=df1.append(df4,ignore_index=True)
print(result)





frames = [df1, df2, df3]

result = pd.concat(frames, keys=['x', 'y', 'z'])
print(result)

print(result.loc['y'])


df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                      'D': ['D2', 'D3', 'D6', 'D7'],
                      'F': ['F2', 'F3', 'F6', 'F7']},
                     index=[2, 3, 6, 7])

result = pd.concat([df1, df4], axis=1)
print(result)

####merging
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
         'A': ['A0', 'A1', 'A2', 'A3'],
          'B': ['B0', 'B1', 'B2', 'B3']})

#print(left)


right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                 'C': ['C0', 'C1', 'C2', 'C3'],
                'D': ['D0', 'D1', 'D2', 'D3']})
#print(right)

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                  'key2': ['K0', 'K1', 'K0', 'K1'],
                   'A': ['A0', 'A1', 'A2', 'A3'],
                  'B': ['B0', 'B1', 'B2', 'B3']})


right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                 'key2': ['K0', 'K0', 'K0', 'K0'],
                  'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']})


result = pd.merge(left, right, on=['key1', 'key2'])
print(result)


result = pd.merge(left, right,on="key",how='right')
print(result)

result = pd.merge(left, right,on="key",how='right')



#########
import pandas as pd
import os
df=pd.read_csv('F:\certification\Pandas-Data-Science-Tasks-master\Pandas-Data-Science-Tasks-master\SalesAnalysis\Sales_Data\Sales_January_2019.csv')
#print(df.shape)

files=os.listdir('F:/freelencing projects/pandas_tutorial/Sales_Data')
#print(files)

files=[file for file in os.listdir('F:/freelencing projects/pandas_tutorial/Sales_Data')]

all_months_data=pd.DataFrame()

for file in files:
    df=pd.read_csv('F:/freelencing projects/pandas_tutorial/Sales_Data/'+file)
    all_months_data=pd.concat([all_months_data,df])
print(all_months_data.head())

all_months_data.to_csv("Megerd.csv")
merged_data=pd.read_csv("Megerd.csv")
print(merged_data.shape)
print(merged_data.tail(100))

#print(merged_data.shape)

nan_df=merged_data[merged_data.isna().any(axis=0)]
print(nan_df.head)

merged_data=merged_data.dropna(how='all')
print(merged_data.head())

merged_data.groupby('Product')



#######
#types of data
#categorical and and numerical
#numerical is of two types discreat and continuous



import pandas as pd
drinks=pd.read_csv('https://bit.ly/drinksbycountry')
print(drinks.head())
#print(drinks.info())
print(drinks.describe())
print(drinks.continent.unique())
drinks['continents']=drinks.continent.astype('category'),inplace=True
print(drinks.continents.cat.codes.head())









#augment data with additional columns


##
import numpy as np
import pandas as pd
frame1 = pd.DataFrame( {'id':['ball','pencil','pen','mug','ashtray'],
  'price': [12.33,11.44,33.21,13.23,33.62]})
print(frame1)

frame2 = pd.DataFrame( {'id':['pencil','pencil','ball','pen'],
                        'color': ['white','red','red','black']})
print(frame2)

print(pd.merge(frame1,frame2))

frame1 = pd.DataFrame( {'id':['ball','pencil','pen','mug','ashtray'],
 'color': ['white','red','red','black','green'],
'brand': ['OMG','ABC','ABC','POD','POD']})
print(frame1)

frame2 = pd.DataFrame( {'id':['pencil','pencil','ball','pen'],
 'brand': ['OMG','POD','ABC','POD']})
print(frame2)

print(pd.merge(frame1,frame2))

print(pd.merge(frame1,frame2,on='id'))

print(pd.merge(frame1,frame2,on='brand'))

frame2.columns = ['brand','sid']
print(frame2)

print(pd.merge(frame1, frame2, left_on='id', right_on='sid'))

frame2.columns=['brand','id']
print(pd.merge(frame1,frame2,on='id',how='outer'))

#merging on index
pd.merge(frame1,frame2,right_index=True, left_index=True)
frame2.columns = ['brand2','id2']
frame1.join(frame2)

frame1 = pd.DataFrame(np.arange(9).reshape(3,3),
index=['white','black','red'],
columns=['ball','pen','pencil'])
print(frame1)

a=np.array([1,2,3])
print(a)

f = np.array([[1, 2, 3],[4, 5, 6]], dtype=complex)
print(f)


###
import numpy

# Discrete Knapsack problem without repetition
def maxGold(W, n, items):
    """ Outputs the maximum weight of gold that fits in knapsack of capacity W
    (int, int, list) -> (int, 2D-array) """

    value = numpy.zeros((W+1, n+1))
    for i in range(1, W+1):
        for j in range(1, n+1):
            # if item i is not part of optimal knapsack
            value[i][j] = value[i][j-1]
            if items[j-1]<=i:
                # if item i is part of optimal knapsack
                temp = value[i-items[j-1]][j-1] + items[j-1]
                # max(i in knapsack, i not in knapsack)
                if temp > value[i][j]:
                    value[i][j] = temp

    return (int(value[W][n]), value)

def printItems(value, items, i, j, arr):
    """ Finds which items are present in optimal solution and returns a boolean array 
    (2D-array, list, int, int, list) -> (list) """

    if i == 0 and j == 0:
        arr.reverse()
        return arr
    if value[i][j] == value[i][j-1]:
        arr.append(0)
        return printItems(value, items, i, j-1, arr)
    else:
        arr.append(1)
        return printItems(value, items, i-items[j-1], j-1, arr)
        
if __name__ == '__main__':
    W, n               = [int(i) for i in input().split()]
    item_weights       = [int(i) for i in input().split()]
    max_weight, Matrix = maxGold(W, n, item_weights)
    bool_vector      = printItems(Matrix, item_weights, W, n, [])
    optimal = [str(j) for i, j in enumerate(item_weights) if bool_vector[i]]
    print(f"Weights in knapsack of capacity {W}: {' '.join(optimal)}")




