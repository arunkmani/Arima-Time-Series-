import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import csv
import warnings 
data1=pd.read_csv('data.txt', sep=' ' , header=None, names=['Date', 'Time','Epoch','ID','Temp','Humidity','Light','Volatge'])
data1['Datetime']=data1['Date']+' '+data1['Time']
data1['Datetime'] = pd.to_datetime(data1['Datetime'])
data1['unixEpoch'] =  data1['Datetime'].apply(lambda x: int(x.timestamp()))
data1.drop(data1[data1['Temp']>100].index, inplace=True)
# data=pd.DataFrame(pd.np.empty((0, 3)))
names=['unixEpoch','Temp','ID']
# data.columns=names
data=data1[names]
data=data[data['ID']==1]
print('-----------------------------------')
#data['Datetime'] = pd.to_datetime(data['Datetime'])
#data= data.drop('ID',axis=1)
data= data.drop('ID',axis=1)
print(data[:5])
print(data.iloc[0]['Temp'])
f1 = open('dataout.csv', 'w')
writer = csv.writer(f1)
for i in range((data.shape[0])):
    row=[data.iloc[i]['unixEpoch'],data.iloc[i]['Temp']]
    writer.writerow(row)
f1.close()
data= data.drop('unixEpoch',axis=1)
# train_data = data[:100]
# test_data = data[101:200]
# #model = sm.tsa.ARIMA(train_data, order=(1,0,0))
# model= ARIMA(train_data,order=(1,0,0))
# model_fit = model.fit()
# predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
# print(type(predictions))
# array1=test_data['Temp']
# array2=predictions.tolist()
# print(data.iloc[210]['Temp'])
i=100
magic=[]
targets=[]
warnings.filterwarnings("ignore", message="ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.")
while(i<(data.shape[0]-100)/1):
    train_data = data[i-100:i]
    test_data = data[i+1:i+100]
    model2= ARIMA(train_data,order=(1,0,0))
    model_fit = model2.fit()
    predictions = model_fit.predict(start=0, end=100 , dynamic=False)
    print(type(predictions))
    #model_fit.save('test_mdl.pkl')
    #print('Predicted:',predictions.iloc[0],'Target:',test_data.iloc[0]['Temp'])
    magic.append(predictions.iloc[0])
    targets.append(test_data.iloc[0]['Temp'])
    i=i+1
separation=0
for i in range(len(magic)):
    difference=abs(magic[i]-targets[i])
    separation=separation+ difference
f1 = open('graph.csv', 'w')
writer = csv.writer(f1)
for i in range(len(magic)):
    row=[magic[i],targets[i]]
    writer.writerow(row)
f1.close()
print('Average Separation is: ',separation/len(magic))
plt.plot(magic, label='Predicted')
plt.plot(targets, label='Actual Values')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Predicted and Actual Values ')
plt.legend()
# Displaying the graph
plt.show()

 