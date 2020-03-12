import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
import statsmodels.api as sm

df =pd.read_csv('perrin-freres-monthly-champagne-.csv')
df.tail()
print(df.tail())
df.drop(106,axis=0,inplace=True)
df.tail()
df.drop(106,axis=0,inplace=True)
df.columns=['Month','Sales Per Month']
print(df.head())
df['Month']=pd.to_datetime(df['Month'])
df.head()
df.set_index('Month',inplace=True)
df.head()
df.plot()


model = sm.tsa.statespace.SARIMAX(df['Sales Per Month'],order=(1,0,0),seasonal_order=(1,1,1,12))
results=model.fit()

df['forecast']=results.predict(start=90,end=130,dynamic=True)
df[['Sales Per Month','forecast']].plot(figsize=(12,8))


from pandas.tseries.offsets import DateOffset
futures_dates =[df.index[-1]+DateOffset(months=x)for x in range(0,24)]
futures_datest_df =pd.DataFrame(index=futures_dates[1:],columns=df.colums)
print(futures_datest_df)
future_df = pd.concat([df,futures_datest_df])
future_df['forecast'] = results.predict(start =104,end = 120,dynamic =True)
future_df[['Sales Per Month','forecast']].plot(figsize=(12,8))
