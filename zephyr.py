import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn.preprocessing import scale
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

df =pd.read_excel('zephyr.xlsx')
print df.columns
'''
Index([u'address', u'district', u'bedroom', u'bath', u'parking', u'sqft',
       u'dollar_sqft', u'sold', u'home_own_ass', u'day_on_market',
       u'list_price', u'sale_price', u'sale_to_list_ratio', u'single_f_h',
       u'condo', u'dist_no'],
      dtype='object')
'''
sold_dt = pd.to_datetime(df.sold, infer_datetime_format=True)
df.sold = sold_dt

model = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + sold + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no', data=df)
linreg = model.fit()
print linreg.summary()

model_nodate = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no', data=df)
linreg2 = model_nodate.fit()
print linreg2.summary()

plt.figure(figsize=(25,15))
plt.plot(df.sort_values('sold').sold, df.sort_values('sold').sale_price)
plt.show()

format = lambda x: x.year
df.sold_year = df.sold.map(format)
df['sold_year'] = pd.Series(df.sold_year, index=df.index)

format2 = lambda x: x.month
df.sold_month = df.sold.map(format2)
df['sold_month'] = pd.Series(df.sold_month, index=df.index)

model_y_m = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no + sold_year + sold_month', data=df)
linreg3 = model_y_m.fit()
print linreg3.summary()

df_address = df.set_index(['address'])
df_sold = df.sort_values('sold')
df_sold = df_sold.set_index('sold')

plt.plot(df_sold.sale_price)
plt.show()
