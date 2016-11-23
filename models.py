import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn.preprocessing import scale, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn import linear_model, svm, metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import f_regression, chi2
import seaborn as sns

df =pd.read_excel('zephyr.xlsx')

sold_dt = pd.to_datetime(df.sold, infer_datetime_format=True)
df.sold = sold_dt

format = lambda x: x.year
df.sold_year = df.sold.map(format)
df['sold_year'] = pd.Series(df.sold_year, index=df.index)

format2 = lambda x: x.month
df.sold_month = df.sold.map(format2)
df['sold_month'] = pd.Series(df.sold_month, index=df.index)

df_sold = df.sort_values('sold')
df_sold = df_sold.set_index('sold')

df_district = pd.get_dummies(df_sold.district)
df_sold_dum = pd.concat( [df_sold, df_district], axis=1)
df_sold_dum.pop('district')

df_sin_month = pd.Series(np.sin(2*np.pi*(df_sold_dum.sold_month)/float(12)))
df_cos_month = pd.Series(np.cos(2*np.pi*(df_sold_dum.sold_month)/float(12)))
df_sold_dum['sin_month'] = pd.Series(df_sin_month, index=df_sold_dum.index)
df_sold_dum['cos_month'] = pd.Series(df_cos_month, index=df_sold_dum.index)

df_sold_dum.pop('sold_month')
df_sold_dum.pop('address')
df_sold_dum.pop('dollar_sqft')
df_sold_dum.pop('sale_to_list_ratio')

y_sold = df_sold_dum.pop('sale_price')
X_sold = df_sold_dum

X_train, X_test, y_train, y_test = train_test_split(X_sold, y_sold, test_size=0.4, random_state=0)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Create a linear regression object
regressor = sm.OLS(y_train,X_train)
regressor = regressor.fit()
#regressor.summary()

params = regressor.params
print 'parameters: \n', params
params =["{0:.4f}".format(x)for x in params]
print '\nparameters: \n', params

pvalues = regressor.pvalues
print '\np-values: \n', pvalues
#pvalues =["{0:.4f}".format(x)for x in pvalues]
#print 'p_values: \n', pvalues
print 'p-values: \n'
for i in range(len(regressor.pvalues)):
    print '{0:.4f}'.format(regressor.pvalues[i])

print 'r-squared: ',regressor.rsquared_adj

'''
Rainbow test for linearity

    The Null hypothesis is that the regression is correctly modelled as linear.
    The alternative for which the power might be large are convex, check

    Parameters
    ----------
    res : Result instance

    Returns
    -------
    fstat : float
        test statistic based of F test
    pvalue : float
        pvalue of the test
'''

sm.stats.linear_rainbow(regressor)

ax = df_sold_dum[['bedroom']].hist(bins = 25)
plt.show()
ax = df_sold_dum[['bath']].hist(bins =12)
plt.show()
ax = df_sold_dum[['sqft']].hist(bins =150)
plt.xlim(0, 10000)
plt.show()

ax = df_sold_dum[['sqft']].hist(bins =150)
plt.xlim(0, 6000)
plt.show()
plt.bar([7, 8, 10],df_sold_dum.groupby('dist_no').count()['bedroom'])
plt.show()
plt.boxplot(df_sold_dum.list_price)
plt.ylim(0, 5000000)
plt.show()

plt.boxplot(df_sold_dum[df_sold_dum.sold_year == 2013].list_price)
plt.ylim(0, 2000000)
plt.show()
plt.boxplot(df_sold_dum[df_sold_dum.sold_year == 2014].list_price)
plt.ylim(0, 2000000)
plt.show()
plt.boxplot(df_sold_dum[df_sold_dum.sold_year == 2015].list_price)
plt.ylim(0, 2000000)
plt.show()
plt.boxplot(df_sold_dum[df_sold_dum.sold_year == 2016].list_price)
plt.ylim(0, 2000000)
plt.show()

every_year = [df[df.sold_year == 2013].sale_price, df[df.sold_year == 2014].sale_price, df[df.sold_year == 2015].sale_price, df[df.sold_year == 2016].sale_price]
plt.boxplot(every_year, positions = [2013, 2014, 2015, 2016])
plt.ylim(0, 2000000)
plt.show()

sns.boxplot(data=df_sold[['list_price', 'sale_price']])
plt.ylim(0, 2000000)
plt.show()

plt.figure(figsize=(25,15))
sns.boxplot(data = every_year)
plt.ylim(0, 2000000)
plt.show()

'''
df_sold.columns
u'address', u'district', u'bedroom', u'bath', u'parking', u'sqft', u'dollar_sqft', u'home_own_ass', u'day_on_market', u'list_price', u'sale_price', u'sale_to_list_ratio', u'single_f_h', u'condo', u'dist_no', u'sold_year', u'sold_month'
'''

model_sold_ym = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no + sold_year + sold_month', data=df_sold)
linreg4 = model_sold_ym.fit()
print linreg4.summary()

model_sold_no_list = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + single_f_h + condo + dist_no + sold_year + sold_month', data=df_sold)
linreg5 = model_sold_no_list.fit()
print linreg5.summary()

model_sold_log = smf.ols(formula='np.log10(sale_price) ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + single_f_h + condo + dist_no + sold_year + sold_month', data=df_sold)
linreg6 = model_sold_log.fit()
print linreg6.summary()

X_sold = df_sold[['district','bedroom','bath','parking','sqft','home_own_ass','day_on_market','list_price','single_f_h','condo','dist_no','sold_year','sold_month']]

d_district = pd.get_dummies(X_sold.district)
X_sold = pd.concat( [X_sold, d_district], axis=1)
X_sold.pop('district')

lr_list_p = linear_model.LinearRegression()
lr_list_p.fit(X_sold, y_sold)

# The coefficients
print'Coefficients: \n', lr_list_p.coef_

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr_list_p.score(X_sold, y_sold))

predicted = cross_val_predict(lr_list_p, X_sold, y_sold, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
ax.scatter(y_sold, predicted)
ax.plot([y_sold.min(), y_sold.max()], [y_sold.min(), y_sold.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

y_sold = np.log10(y_sold)

lr_list_2 = linear_model.LinearRegression()
lr_list_2.fit(X_sold, y_sold)

# The coefficients
print'Coefficients: \n', lr_list_2.coef_

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr_list_2.score(X_sold, y_sold))

predicted = cross_val_predict(lr_list_2, X_sold, y_sold, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
ax.scatter(y_sold, predicted)
ax.plot([y_sold.min(), y_sold.max()], [y_sold.min(), y_sold.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
