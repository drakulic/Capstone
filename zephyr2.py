from __future__ import division
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
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, svm, metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import f_regression, chi2
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import warnings

warnings.filterwarnings('ignore')

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
fig.savefig('sale.png')

sp_log = np.log10(df.sort_values('sold').sale_price)

fig = plt.figure(figsize=(25,15))
plt.plot(df.sort_values('sold').sold, sp_log)
fig.suptitle('log10 of sale price over time', fontsize=20)
plt.xlabel('Sold', fontsize=20)
plt.ylabel('log10 of sale price', fontsize=20)
plt.show()
fig.savefig('log10_sale.png')

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
print
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
fig.savefig('log10_list.png')

fig = plt.figure(figsize=(25,15))
plt.plot(df_sold[['list_price']], color = 'blue')
plt.plot(df_sold[['sale_price']], color = 'red')
plt.xlabel('Sold', fontsize=20)
fig.suptitle('sale and list price over time', fontsize=20)
plt.ylabel('sale red,     list blue', fontsize=20)
plt.show()
fig.savefig('sale_list.png')

fig = plt.figure(figsize=(25,15))
plt.plot(np.log10(df_sold.sale_price), color='red')
plt.plot(np.log10(df_sold.list_price), color = 'blue')
plt.xlabel('Sold', fontsize=20)
fig.suptitle('log10 of sale and list price over time', fontsize=20)
plt.ylabel('sale red,     list blue', fontsize=20)
plt.show()
fig.savefig('log10_sale_list.png')


fig = plt.figure(figsize=(25,15))
plt.scatter(df_sold.index, np.log10(df_sold.sale_price))
fig.suptitle('sale price over time', fontsize=20)
plt.ylabel('log10 of sale price', fontsize=20)
plt.show()
fig.savefig('log10_sale_scatter.png')

fig = plt.figure(figsize=(25,15))
plt.scatter(df_sold.index, np.log10(df_sold.list_price))
fig.suptitle('list price over time', fontsize=20)
plt.ylabel('log10 of list price', fontsize=20)
plt.show()
fig.savefig('log10_list_scatter.png')
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
#print 'lr1 no log10 Train Coefficients: \n', lr1.coef_
# Explained variance score: 1 is perfect prediction
print
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
fig.savefig('LR_w_list_Train.png')

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
fig.savefig('LR_w_list_Test.png')

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
fig.savefig('LR_w_list_Validation.png')

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
plt.show()
fig.savefig('residuals.png')

fig, ax = plt.subplots(figsize = (25,15))
sm.graphics.qqplot(resids, line='s', fit=True, ax =ax)
plt.show()
fig.savefig('qqplot.png')

#smg.influence_plot(linreg4)
print
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
#print 'lr2 Train Coefficients: \n', lr2.coef_
# Explained variance score: 1 is perfect prediction
print
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
fig.savefig('LR2_train.png')

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
fig.savefig('LR2_test.png')

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
#print 'no list price Train Coefficients: \n', lr3.coef_
# Explained variance score: 1 is perfect prediction
print
print('lr3 no list price Train Variance score: %.2f' % lr3.score(X_train, y_train))

# Train
predicted_train3 = cross_val_predict(lr3, X_train, y_train, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr3 LinearRegression without list price', fontsize=20)
ax.scatter(y_train, predicted_train3)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('lr3 Train set Measured', fontsize=20)
ax.set_ylabel('lr3 Train set Predicted', fontsize=20)
plt.show()
fig.savefig('LR3_no_list_train.png')

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
fig.savefig('LR3_no_list_test.png')

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
fig.savefig('Whole_no_list.png')

print
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
print
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
print
print('lr_log Train Variance score: %.2f' % lr_log.score(X_train, y_train_log))
predicted_train_log = cross_val_predict(lr_log, X_train, y_train_log, cv=10)

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('lr_log LinearRegression without list price', fontsize=20)
ax.scatter(y_train_log, predicted_train_log)
ax.plot([y_train_log.min(), y_train_log.max()], [y_train_log.min(), y_train_log.max()], 'k--', lw=4)
ax.set_xlabel('lr_log Train set Measured', fontsize=20)
ax.set_ylabel('lr_log Train set Predicted', fontsize=20)
plt.show()
fig.savefig('log10_no_list_train.png')

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
fig.savefig('log10_no_list_test.png')

print
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
fig.savefig('log10_no_list_whole.png')

print
print('lr_log_all All Variance score: %.2f' % lr_log_all.score(X_sold, y_sold_log))

print
print 'Date        Square-Foot     Sale-price    Predicted price '
for i in xrange(30):
    print X_test.index[i].date(),'     ', X_test.sqft[i],'     ', np.power(10, y_test_log[i]), '     ', np.power(10, predicted_all_log[4000+i]).round(2)
print
print
###############################################
'''
rf = RandomForest(num_trees=100, num_features=10)
rf.fit(X_train, np.log10(y_train))
print "Random Forest score:", rf.score(X_test, np.log10(y_test))
'''
###### use X_sold, y_sold_log
train_x = X_sold[:4000]
test_x = X_sold[4000:]

train_y = y_sold_log[:4000]
test_y = y_sold_log[4000:]

col_names = X_sold.columns

def cross_val(estimator, train_x, train_y):
    # n_jobs=-1 uses all the cores on your machine
    mse = cross_val_score(estimator, train_x, train_y,
                           scoring='mean_squared_error',
                           cv=10, n_jobs=-1) * -1
    r2 = cross_val_score(estimator, train_x, train_y,
                           scoring='r2', cv=10, n_jobs=-1)

    mean_mse = mse.mean()
    mean_r2 = r2.mean()

    params = estimator.get_params()
    name = estimator.__class__.__name__
    print '%s Train CV | MSE: %.3f | R2: %.3f' % (name, mean_mse, mean_r2)
    return mean_mse, mean_r2

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=1)

gdbr = GradientBoostingRegressor(learning_rate=0.1, loss='ls',
                                 n_estimators=100, random_state=1)

abr = AdaBoostRegressor(DecisionTreeRegressor(), learning_rate=0.1,
                        loss='linear', n_estimators=100, random_state=1)

mean_mse, mean_r2 = cross_val(rf, train_x, train_y)
#print '%s Train CV | MSE: %.3f | R2: %.3f' % (RandomForestRegressor, mean_mse, mean_r2)
mean_mse, mean_r2 = cross_val(gdbr, train_x, train_y)
#print '%s Train CV | MSE: %.3f | R2: %.3f' % (GradientBoostingRegressor, mean_mse, mean_r2)
mean_mse, mean_r2 = cross_val(abr, train_x, train_y)
#print '%s Train CV | MSE: %.3f | R2: %.3f' % (AdaBoostRegressor, mean_mse, mean_r2)

rf.fit(train_x, train_y)
gdbr.fit(train_x, train_y)
abr.fit(train_x, train_y)

def rf_score_plot():
    rf_test_y_pred = rf.predict(test_x)
    test_mse = mean_squared_error(rf_test_y_pred, test_y)
    plt.axhline(test_mse, alpha=.7, c='y' , lw=3, ls='-.', label='Random Forest Test')

def stage_score_plot(estimator, train_x, train_y, test_x, test_y):
    name = estimator.__class__.__name__.replace('Regressor', '')
    learn_rate = estimator.learning_rate

    train_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)

    # Get train score from each boost
    for i, train_y_pred in enumerate(estimator.staged_predict(train_x)):
        train_scores[i] = mean_squared_error(train_y, train_y_pred)

    # Get test score from each boost
    for i, test_y_pred in enumerate(estimator.staged_predict(test_x)):
        test_scores[i] = mean_squared_error(test_y, test_y_pred)

    plt.plot(train_scores, alpha=.5, label='%s Train - Rate %s' % (name, learn_rate))
    plt.plot(test_scores, alpha=.5, label='%s Test - Rate %s' % (name, learn_rate), ls='--')
    plt.title(name, fontsize=16, fontweight='bold')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)

fig = plt.figure(figsize=(25,15))
stage_score_plot(gdbr, train_x, train_y, test_x, test_y)
rf_score_plot()
plt.legend(loc='best')
plt.show()
fig.savefig('gdbr.png')

fig = plt.figure(figsize=(25,15))
stage_score_plot(abr, train_x, train_y, test_x, test_y)
rf_score_plot()
plt.legend(loc='best')
plt.show()
fig.savefig('abr.png')

abr_high_learn = AdaBoostRegressor(learning_rate=1, loss='linear',
                                           n_estimators=100, random_state=1)

gdbr_high_learn = GradientBoostingRegressor(learning_rate=1, loss='ls',
                                           n_estimators=100, random_state=1)

abr_high_learn.fit(train_x, train_y)

fig = plt.figure(figsize=(25,15))
stage_score_plot(abr, train_x, train_y, test_x, test_y)
stage_score_plot(abr_high_learn, train_x, train_y, test_x, test_y)
plt.legend(loc='best')
plt.show()
fig.savefig('abr_best.png')

gdbr_high_learn.fit(train_x, train_y)

fig = plt.figure(figsize=(25,15))
stage_score_plot(gdbr, train_x, train_y, test_x, test_y)
stage_score_plot(gdbr_high_learn, train_x, train_y, test_x, test_y)
plt.legend(loc='best')
plt.show()
fig.savefig('gdbr_best.png')

rf_grid = {'max_depth': [3, None],
           'max_features': [1, 3, 10],
           'min_samples_split': [1, 3, 10],
           'min_samples_leaf': [1, 3, 10],
           'bootstrap': [True, False],
           'n_estimators': [25, 40, 50],
           'random_state': [1]}

gd_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
           'max_depth': [4, 6],
           'min_samples_leaf': [3, 5, 9, 17],
           'max_features': [1.0, 0.3, 0.1],
           'n_estimators': [500],
           'random_state': [1]}

def grid_search(est, grid):
    grid_cv = GridSearchCV(est, grid, n_jobs=-1, verbose=True,
                           scoring='mean_squared_error').fit(train_x, train_y)
    return grid_cv
print
print 'Grid search RandomForestRegressor'
rf_grid_search = grid_search(RandomForestRegressor(), rf_grid)
print
print 'Grid search GradientBoostingRegressor'
gd_grid_search = grid_search(GradientBoostingRegressor(), gd_grid)

print
rf_best = rf_grid_search.best_estimator_
print 'RandomForestRegressor best estimator: \n'
print rf_best
print
print
print 'GradientBoostingRegressor best estimator: \n'
gd_best = gd_grid_search.best_estimator_
print gd_best
print

gd_grid_search.best_params_
rf_grid_search.best_params_

cross_val(gd_best, train_x, train_y)
cross_val(rf_best, train_x, train_y)

print
test_ypred = gd_best.predict(test_x)
print 'Gradient Boost Test MSE:', mean_squared_error(test_ypred, test_y)
print 'Gradient Boost Test R2:',r2_score(test_ypred, test_y)
print
test_ypred = rf_best.predict(test_x)
print 'Random Forest Test MSE:', mean_squared_error(test_ypred, test_y)
print 'Random Forest Test R2:', r2_score(test_ypred, test_y)

# sort importances
indices = np.argsort(gd_best.feature_importances_)
# plot as bar chart
figure = plt.figure(figsize=(25,15))
plt.barh(np.arange(len(col_names)), gd_best.feature_importances_[indices],
         align='center', alpha=.5)
plt.yticks(np.arange(len(col_names)), np.array(col_names)[indices], fontsize=14)
plt.xticks(fontsize=14)
_ = plt.xlabel('Relative importance', fontsize=18)
plt.show()
fig.savefig('feature_importances.png')

fig, axs = plot_partial_dependence(gd_best, train_x, range(X_sold.shape[1]) ,
                                   feature_names=col_names, figsize=(25,15))
fig.tight_layout()
plt.show()
fig.savefig('dependence.png')

#################
# The best model
gdbr_best = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.02, loss='ls', max_depth=6, max_features=0.3,
             max_leaf_nodes=None, min_impurity_split=1e-07,
             min_samples_leaf=5, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=500,
             presort='auto', random_state=1, subsample=1.0, verbose=0,
             warm_start=False)

gdbr_best.fit(train_x, train_y)

fig = plt.figure(figsize=(25,15))
stage_score_plot(gdbr_best, train_x, train_y, test_x, test_y)
plt.legend(loc='best')
plt.show()
fig.savefig('final_model.png')

print
test_ypred_b = gdbr_best.predict(test_x)
print 'Gradient Boost Test MSE:', mean_squared_error(test_ypred_b, test_y)
print 'Gradient Boost Test R2:',r2_score(test_ypred_b, test_y)
print
print
print 'Date        Square-Foot     Sale-price    Predicted price '
for i in xrange(30):
    print test_x.index[i].date(),'     ', test_x.sqft[i],'     ', np.power(10, test_y[i]), '     ', np.power(10, test_ypred_b[i]).round(2)
print
print

fig, ax = plt.subplots(figsize=(25,15))
fig.suptitle('Final model without list price, GradientBoostingRegressor', fontsize=20)
ax.scatter(test_y, test_ypred_b)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('GB log10 test set Measured', fontsize=20)
ax.set_ylabel('GB log10 test set Predicted', fontsize=20)
plt.show()
fig.savefig('final_GB.png')


#################
sale_avg = np.df_sold.sale_price.mean()
with_list_avg = np.around(np.power(10, linreg3.fittedvalues), decimals = 2).mean()
without_list_avg = np.around(np.power(10, linreg4.fittedvalues), decimals = 2).mean()
s_w = sale_avg - with_list_avg
s_wo = sale_avg - without_list_avg
gb_s_avg = np.power(10, test_y).mean()
gb_pred_avg = np.power(10, test_ypred_b).round(2).mean()

print
print 'sale price avg: ', sale_avg
print '\nwith list price avg: ', with_list_avg
print '\ndiff: ', s_w
print
print 'without list price avg: ', without_list_avg
print '\ndiff: ', s_wo
print
print 'sale price avg for GB test', gb_s_avg
print '\nGB predicted avg: ', gb_pred_avg
print '\ndiff: ', gb_s_avg - gb_pred_avg
