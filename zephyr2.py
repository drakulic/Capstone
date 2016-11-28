import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics import regressionplots as smg
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
from sklearn.metrics import mean_squared_error


df =pd.read_excel('zephyr_no_outl.xlsx')

'''
print df.columns
Index([u'address', u'district', u'bedroom', u'bath', u'parking', u'sqft',
       u'dollar_sqft', u'sold', u'home_own_ass', u'day_on_market',
       u'list_price', u'sale_price', u'sale_to_list_ratio', u'single_f_h',
       u'condo', u'dist_no'],
      dtype='object')
'''
sold_dt = pd.to_datetime(df.sold, infer_datetime_format=True)
df.sold = sold_dt

'''
print 'model with list price: \n'
model = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + sold + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no', data=df)
linreg = model.fit()
print linreg.summary()
#print linreg.params
'''
'''
print '\nmodel without date: \n'
model_nodate = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no', data=df)
linreg2 = model_nodate.fit()
print linreg2.summary()
#print linreg2.params
'''

fig = plt.figure(figsize=(25,15))
plt.plot(df.sort_values('sold').sold, df.sort_values('sold').sale_price)
fig.suptitle('sale price over time', fontsize=20)
plt.xlabel('Sold', fontsize=20)
plt.ylabel('sale price', fontsize=20)
plt.show()

sp_log = np.log10(df.sort_values('sold').sale_price)

fig = plt.figure(figsize=(25,15))
plt.plot(df.sort_values('sold').sold, sp_log)
fig.suptitle('log10 of sale price over time', fontsize=20)
plt.xlabel('Sold', fontsize=20)
plt.ylabel('log10 of sale price', fontsize=20)
plt.show()

format = lambda x: x.year
df.sold_year = df.sold.map(format)
df['sold_year'] = pd.Series(df.sold_year, index=df.index)

format2 = lambda x: x.month
df.sold_month = df.sold.map(format2)
df['sold_month'] = pd.Series(df.sold_month, index=df.index)


print '\nmodel with year month columns, no log10: \n'
model_y_m = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no + sold_year + sold_month', data=df)
linreg_no_log = model_y_m.fit()
print linreg_no_log.summary()
#print linreg3.params


df_address = df.set_index(['address'])
df_sold = df.sort_values('sold')
df_sold = df_sold.set_index('sold')

'''
plt.figure(figsize=(25,15))
plt.plot(df_sold.sale_price)
plt.show()
'''
'''
fig = plt.figure(figsize=(25,15))
plt.plot(df_sold[['list_price']], color = 'green')
plt.plot(df_sold[['sale_price']], color = 'red')
plt.xlabel('Sold', fontsize=20)
fig.suptitle('sale and list price over time', fontsize=20)
plt.ylabel('sale red,     list green', fontsize=20)
plt.show()
'''
fig = plt.figure(figsize=(25,15))
np.log10(df_sold.list_price).plot()
fig.suptitle('log10 of list price over time', fontsize=20)
plt.xlabel('Sold', fontsize=20)
plt.ylabel('log10 of list price', fontsize=20)
plt.show()

fig = plt.figure(figsize=(25,15))
plt.plot(df_sold[['list_price']], color = 'blue')
plt.plot(df_sold[['sale_price']], color = 'red')
plt.xlabel('Sold', fontsize=20)
fig.suptitle('sale and list price over time', fontsize=20)
plt.ylabel('sale red,     list blue', fontsize=20)
plt.show()

fig = plt.figure(figsize=(25,15))
plt.plot(np.log10(df_sold.sale_price), color='red')
plt.plot(np.log10(df_sold.list_price), color = 'blue')
plt.xlabel('Sold', fontsize=20)
fig.suptitle('log10 of sale and list price over time', fontsize=20)
plt.ylabel('sale red,     list blue', fontsize=20)
plt.show()


fig = plt.figure(figsize=(25,15))
plt.scatter(df_sold.index, np.log10(df_sold.sale_price))
fig.suptitle('sale price over time', fontsize=20)
plt.ylabel('log10 of sale price', fontsize=20)
plt.show()

fig = plt.figure(figsize=(25,15))
plt.scatter(df_sold.index, np.log10(df_sold.list_price))
fig.suptitle('list price over time', fontsize=20)
plt.ylabel('log10 of list price', fontsize=20)
plt.show()
#print 'listed for > $5M:', df_sold.list_price[df_sold.list_price > 5000000].count()
#print 'sold for > $5M:', df_sold.sale_price[df_sold.sale_price > 5000000].count()
'''
fig = plt.figure(figsize=(25,15))
np.log10(df_sold.sale_price).plot()
fig.suptitle('log10 of sale price over time', fontsize=20)
plt.ylabel('log10 of sale price', fontsize=20)
plt.show()
'''



'''
print '\nmodel with year month columns and list price: \n'
model_y_m = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no + sold_year + sold_month', data=df_sold)
linreg4 = model_y_m.fit()
print linreg4.summary()
print linreg4.fittedvalues[4000:4031]
#print linreg3.params
'''

# Cross-validationc
df_district = pd.get_dummies(df_sold.district)
df_sold_dum = pd.concat( [df_sold, df_district], axis=1)
df_sold_dum.pop('district')
df_sold_dum.pop('address')

y_sold = df_sold_dum.pop('sale_price')
#y_sold = np.log10(y_sold)

X_sold = df_sold_dum
X_sold.pop('dollar_sqft')
X_sold.pop('sale_to_list_ratio')

#X_train, X_test, y_train, y_test = train_test_split(X_sold, y_sold, test_size=0.3, random_state=0)
X_train = X_sold[:3500]
X_test = X_sold[3500:4000]
X_val = X_sold[4000:]

y_train = y_sold[:3500]
y_test = y_sold[3500:4000]
y_val = y_sold[4000:]

'''
X_train.shape
X_test.shape
X_val.shape
y_train.shape
y_test.shape
y_val.shape
'''
#############################
# Train

lr1 = linear_model.LinearRegression()
lr1.fit(X_train, y_train)

# The coefficients
print 'lr1 no log10 Train Coefficients: \n', lr1.coef_
# Explained variance score: 1 is perfect prediction
print('lr1 no log10 Train Variance score: %.2f' % lr1.score(X_train, y_train))

# Train
predicted_t1 = cross_val_predict(lr1, X_train, y_train, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr1 no log10 LinearRegression with list price', fontsize=20)
ax.scatter(y_train, predicted_t1)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('lr1 no log10 Train set Measured', fontsize=20)
ax.set_ylabel('lr1 no log10 Train set Predicted', fontsize=20)
plt.show()

###################################
# Test

# The coefficients
#print 'Test Coefficients: \n', lr1.coef_
# Explained variance score: 1 is perfect prediction
print('lr1 no log10 Test Variance score: %.2f' % lr1.score(X_test, y_test))

# Train
predicted_test1 = cross_val_predict(lr1, X_test, y_test, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr1 no log10 LinearRegression with list price', fontsize=20)
ax.scatter(y_test, predicted_test1)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('lr1 no log10 Test set Measured', fontsize=20)
ax.set_ylabel('lr1 no log10 Test set Predicted', fontsize=20)
plt.show()

###################################
# Validation

# The coefficients
#print 'Validation Coefficients: \n', lr1.coef_
# Explained variance score: 1 is perfect prediction
print('lr1 no log10 Validation Variance score: %.2f' % lr1.score(X_val, y_val))

# Train
predicted_val = cross_val_predict(lr1, X_val, y_val, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr1 no log10 LinearRegression with list price', fontsize=20)
ax.scatter(y_val, predicted_val)
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
ax.set_xlabel('lr1 no log10 Validation set Measured', fontsize=20)
ax.set_ylabel('lr1 no log10 Validation set Predicted', fontsize=20)
plt.show()

model_sold_list = smf.ols(formula='np.log10(sale_price) ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + np.log10(list_price) + single_f_h + condo + dist_no + sold_year + sold_month', data=df_sold)
linreg3 = model_sold_list.fit()
print '\nmodel with log10 list price: ', linreg3.summary()

model_sold_no_list= smf.ols(formula='np.log10(sale_price) ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + single_f_h + condo + dist_no + sold_year + sold_month', data=df_sold)
linreg4 = model_sold_no_list.fit()
print '\nmodel without list price: ', linreg4.summary()

resids = linreg4.outlier_test()['student_resid']

#residual plot 1
fig = plt.figure(figsize=(25,15))
plt.plot(linreg4.fittedvalues, resids, 'o')
plt.xlabel('studentized residuals', fontsize=20)
plt.ylabel('predicted response', fontsize=20)
plt.axhline(0, c='r', linestyle = '--')

sm.graphics.qqplot(resids, line='s', fit=True)
smg.influence_plot(linreg4)

print 'Date        Sale price  Pred. with list p    W/o list p     W - W/o diff'
for i in xrange(30):
    print df_sold.index[4000+i].date(), '    ', df_sold.sale_price[4000+i], '    ', np.around(np.power(10, linreg3.fittedvalues[4000+i]), decimals = 2), '     ', np.around(np.power(10, linreg4.fittedvalues[4000+i]), decimals = 2), '    ',np.around(np.power(10,linreg3.fittedvalues[4000+i])-np.power(10,linreg4.fittedvalues[4000+i]), decimals = 2)


#print 'Date        Predicted w/o list price     S-P error'
#print np.power(10, linreg4.fittedvalues[4000:4031])


'''
f, pval = f_regression(X_train, y_train, center=False)
print '\nTrain p-values: ', pval

f, pval = f_regression(X_test, y_test, center=False)
print 'Test p-values: ', pval

f, pval = f_regression(X_val, y_val, center=False)
print 'Validation p-values: ', pval
'''
#######################################
# different train-test split
#######################################
X_train = X_sold[:4000]
X_test = X_sold[4000:]

y_train = y_sold[:4000]
y_test = y_sold[4000:]

#############################
# Train

lr2 = linear_model.LinearRegression()
lr2.fit(X_train, y_train)

# The coefficients
print 'lr2 Train Coefficients: \n', lr2.coef_
# Explained variance score: 1 is perfect prediction
print('lr2 Train Variance score: %.2f' % lr2.score(X_train, y_train))

# Train
predicted_train2 = cross_val_predict(lr2, X_train, y_train, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr2 LinearRegression with list price', fontsize=20)
ax.scatter(y_train, predicted_train2)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('lr2 Train set Measured', fontsize=20)
ax.set_ylabel('lr2 Train set Predicted', fontsize=20)
plt.show()

###################################
# Test

# The coefficients
#print 'Test Coefficients: \n', lr2.coef_
# Explained variance score: 1 is perfect prediction
print('lr2 Test Variance score: %.2f' % lr2.score(X_test, y_test))

# Train
predicted_test2 = cross_val_predict(lr2, X_test, y_test, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr2 LinearRegression with list price', fontsize=20)
ax.scatter(y_test, predicted_test2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('lr2 Test set Measured', fontsize=20)
ax.set_ylabel('lr2 Test set Predicted', fontsize=20)
plt.show()

###################################
'''
f, pval = f_regression(X_train, y_train, center=False)
print '\nTrain p-values: ', pval

f, pval = f_regression(X_test, y_test, center=False)
print 'Test p-values: ', pval
'''
###################################
# No list price
###################################
X_train.pop('list_price')
X_test.pop('list_price')
# Train

lr3 = linear_model.LinearRegression()
lr3.fit(X_train, y_train)

# The coefficients
print 'no list price Train Coefficients: \n', lr3.coef_
# Explained variance score: 1 is perfect prediction
print('no list price Train Variance score: %.2f' % lr3.score(X_train, y_train))

# Train
predicted_train3 = cross_val_predict(lr3, X_train, y_train, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr3 LinearRegression without list price', fontsize=20)
ax.scatter(y_train, predicted_train3)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('lr3 Train set Measured', fontsize=20)
ax.set_ylabel('lr3 Train set Predicted', fontsize=20)
plt.show()

###################################
# Test

# The coefficients
#print 'Test Coefficients: \n', lr3.coef_
# Explained variance score: 1 is perfect prediction
print('lr3 Test Variance score: %.2f' % lr3.score(X_test, y_test))

# Train
predicted_test3 = cross_val_predict(lr3, X_test, y_test, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr3 LinearRegression without list price', fontsize=20)
ax.scatter(y_test, predicted_test3)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('lr3 Test set Measured', fontsize=20)
ax.set_ylabel('lr3 Test set Predicted', fontsize=20)
plt.show()

# The whole set
X_sold.pop('list_price')
predicted_all = cross_val_predict(lr3, X_sold, y_sold, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('whole set LinearRegression without list price', fontsize=20)
ax.scatter(y_sold, predicted_all)
ax.plot([y_sold.min(), y_sold.max()], [y_sold.min(), y_sold.max()], 'k--', lw=4)
ax.set_xlabel('Measured', fontsize=20)
ax.set_ylabel('Predicted', fontsize=20)
plt.show()

print('lr3 All Variance score: %.2f' % lr3.score(X_sold, y_sold))
###################################
'''
f, pval = f_regression(X_train, y_train, center=False)
print 'Train p-values: ', pval

f, pval = f_regression(X_test, y_test, center=False)
print 'Test p-values: ', pval

f, pval = f_regression(X_sold, y_sold, center=False)
print 'The whole set p-values: ', pval

mse = mean_squared_error(y_sold, predicted_all)
#mse2 = mean_squared_error(np.power(10, y_sold), np.power(10, predicted_all))
rmse = np.sqrt(mse)
#print 'mean_squared_error for log10: ', mse
print 'mse for predicted sale price', mse
print 'root-mean-square error: ', rmse
'''
'''
print 'Date        Square-Foot     Sale-price    Predicted price w LP   Predicted w/o LP    S-P error'
for i in xrange(30):
    print X_test.index[i].date(),'     ', X_test.sqft[i],'     ', np.power(10, y_test[i]), '     ', np.power(10, predicted_test2[i]), '     ', np.power(10, predicted_test3[i]), '    ', np.power(10,  y_test[i])-np.power(10, predicted_test2[i])
'''
'''
print 'Date        Square-Foot     Sale-price    Predicted price w LP   Predicted w/o LP    S-P error'
for i in xrange(30):
    print X_test.index[i].date(),'     ', X_test.sqft[i],'     ', y_test[i], '     ', predicted_test2[i], '     ', predicted_test3[i], '    ', y_test[i]-predicted_test2[i]
'''
print 'Date        Square-Foot     Sale-price    Predicted price w LP   Predicted w/o LP    S-P error'
for i in xrange(30):
    print X_test.index[i].date(),'     ', X_test.sqft[i],'     ', y_test[i], '     ', np.round(predicted_test2[i], 2), '     ', np.round(predicted_all[i], 2), '    ', np.round(y_test[i]-predicted_all[i], 2)
#######################################
# with log10 of sale price
y_train_log = np.log10(y_train)
y_test_log = np.log10(y_test)

lr_log= linear_model.LinearRegression(normalize = True)
lr_log.fit(X_train, y_train_log)
# Train
print('lr_log Train Variance score: %.2f' % lr_log.score(X_train, y_train_log))
predicted_train_log = cross_val_predict(lr_log, X_train, y_train_log, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr_log LinearRegression without list price', fontsize=20)
ax.scatter(y_train_log, predicted_train_log)
ax.plot([y_train_log.min(), y_train_log.max()], [y_train_log.min(), y_train_log.max()], 'k--', lw=4)
ax.set_xlabel('lr_log Train set Measured', fontsize=20)
ax.set_ylabel('lr_log Train set Predicted', fontsize=20)
plt.show()

###################################
# Test

# The coefficients
#print 'Test Coefficients: \n', lr3.coef_
# Explained variance score: 1 is perfect prediction
print('lr_log Test Variance score: %.2f' % lr_log.score(X_test, y_test_log))

# Train
predicted_test_log = cross_val_predict(lr_log, X_test, y_test_log, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr_log LinearRegression without list price', fontsize=20)
ax.scatter(y_test_log, predicted_test_log)
ax.plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'k--', lw=4)
ax.set_xlabel('lr_log Test set Measured', fontsize=20)
ax.set_ylabel('lr_log Test set Predicted', fontsize=20)
plt.show()

print 'Date        Square-Foot     Sale-price    Predicted price '
for i in xrange(30):
    print X_test.index[i].date(),'     ', X_test.sqft[i],'     ', np.power(10, y_test_log[i]), '     ', np.power(10, predicted_test_log[i]).round(2)

##########################################
# The whole set

y_sold_log = np.log10(y_sold)

lr_log_all= linear_model.LinearRegression(normalize = True)
lr_log_all.fit(X_sold, y_sold_log)
predicted_all_log = cross_val_predict(lr_log_all, X_sold, y_sold_log, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('whole set log10 price LinearRegression without list price', fontsize=20)
ax.scatter(y_sold_log, predicted_all_log)
ax.plot([y_sold_log.min(), y_sold_log.max()], [y_sold_log.min(), y_sold_log.max()], 'k--', lw=4)
ax.set_xlabel('Measured', fontsize=20)
ax.set_ylabel('Predicted', fontsize=20)
plt.show()

print('lr_log_all All Variance score: %.2f' % lr_log_all.score(X_sold, y_sold_log))

print 'Date        Square-Foot     Sale-price    Predicted price '
for i in xrange(30):
    print X_test.index[i].date(),'     ', X_test.sqft[i],'     ', np.power(10, y_test_log[i]), '     ', np.power(10, predicted_all_log[4000+i]).round(2)

###############################################
