import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import patsy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from scipy import stats
import seaborn as sns
import datetime

start = datetime.datetime(2018,1,1)
end = datetime.datetime(2015,7,7)

from pandas_datareader import data, web
import fix_yahoo_finance as yf
yf.pdr_override()
twtr =web. DataReader("TWTR","yahoo",start,end)
google =web.DataReader("GOOG","yahoo",start,end)
twtr.head()
google.head()

close_twtr = twtr["Close"]
close_goggle = google["Close"]
close_twtr.describe()
close_google.describe()

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
close_twtr.plot(ax=ax[0])
ax[0].set_title("TWTR")
close_google.plot(ax=ax[1])
ax[1].set_title("GOOG")
plt.show()
 

stock = pd.merge(twtr,google,left_index = True, right_index = True)
stock = stock[["Close_x","Close_y"]]
stock.columns = ["TWTR","GOOG"]
stock.head()

daily_return = (stock.diff()/stock.shift(periods = 1)).dropna()
daily_return.head()
 
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
daily_return["TWTR"].plot(ax=ax[0])
ax[0].set_title("TWTR")
daily_return["GOOG"].plot(ax=ax[1])
ax[1].set_title("GOOG")
plt.show()

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
sns.distplot(daily_return["TWTR"],ax=ax[0])
ax[0].set_title("TWTR")
sns.distplot(daily_return["GOOG"],ax=ax[1])
ax[1].set_title("GOOG")
plt.show() 

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
plt.scatter(daily_return["GOOG"],daily_return["TWTR"])
plt.title("Scatter Plot of daily return between GOOG and TWTR")
plt.show()
 
import statsmodels.api as sm
daily_return["intercept"]=1.0
model = sm.OLS(daily_return["GOOG"],daily_return[["TWTR","intercept"]])
results = model.fit()
results.summary()
 