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
#linreg = model.fit()
#print linreg.summary()
#print linreg.params

model_nodate = smf.ols(formula='sale_price ~ district + bedroom + bath + parking + sqft + home_own_ass + day_on_market + list_price + single_f_h + condo + dist_no', data=df)
linreg2 = model_nodate.fit()
#print linreg2.summary()
#print linreg2.params

plt.figure(figsize=(25,15))
plt.plot(df.sort_values('sold').sold, df.sort_values('sold').sale_price)
plt.show()

plt.figure(figsize=(25,15))
plt.plot(df.sort_values(['sold', 'dist_no']).sold, df.sort_values(['sold', 'dist_no']).sale_price)
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
#print linreg3.params

df_address = df.set_index(['address'])
df_sold = df.sort_values('sold')
df_sold = df_sold.set_index('sold')

plt.figure(figsize=(25,15))
plt.plot(df_sold.sale_price)
plt.show()

plt.figure(figsize=(25,15))
plt.plot(df_sold.sale_price, color='red')
plt.plot(df_sold.list_price, color = 'blue')
plt.show()

plt.figure(figsize=(25,15))
plt.plot(df_sold[['list_price']], color = 'green')
plt.plot(df_sold[['sale_price']], color = 'red')
plt.show()

plt.figure(figsize=(25,15))
plt.scatter(df_sold.index, (np.log10(df_sold.sale_price)))
plt.show()

print 'listed for > $5M:', df_sold.list_price[df_sold.list_price > 5000000].count()
print 'sold for > $5M:', df_sold.sale_price[df_sold.sale_price > 5000000].count()

plt.figure(figsize=(25,15))
df_sold.sale_price.plot()
plt.show()

plt.figure(figsize=(25,15))
df_sold.list_price.plot()
plt.show()

# Cross-validation
y_sold = df_sold.pop('sale_price')
X_sold = df_sold[['district','bedroom','bath','parking','sqft','home_own_ass','day_on_market','list_price','single_f_h','condo','dist_no','sold_year','sold_month']]

X_train, X_test, y_train, y_test = train_test_split(X_sold, y_sold, test_size=0.4, random_state=0)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

X_train.pop('district')
X_test.pop('district')

lr1 = linear_model.LinearRegression()
lr1.fit(X_train, y_train)

# The coefficients
print('Coefficients: \n', lr1.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((lr1.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr1.score(X_test, y_test))
'''
# Plot outputs
plt.scatter(X_test, y_test,  color='black', figsize=(25,15))
plt.plot(X_test, lr1.predict(X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
'''
predicted = cross_val_predict(lr1, X_train, y_train, cv=10)
#print('metrics accuracy_score: ', metrics.accuracy_score(y_train, predicted))

fig, ax = plt.subplots(figsize=(25,15))
ax.scatter(y_train, predicted)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

'''
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''

'''
clf = SVC()
clf.fit(X, y)

Methods
decision_function(X)	Distance of the samples X to the separating hyperplane.
fit(X, y[, sample_weight])	Fit the SVM model according to the given training data.
get_params([deep])	Get parameters for this estimator.
predict(X)	Perform classification on samples in X.
score(X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels.
set_params(\*\*params)	Set the parameters of this estimator.

It is also possible to use other cross validation strategies by passing a cross validation iterator instead, for instance:
>>>
>>> from sklearn.model_selection import ShuffleSplit
>>> n_samples = iris.data.shape[0]
>>> cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
>>> cross_val_score(clf, iris.data, iris.target, cv=cv)

Data transformation with held out data
Just as it is important to test a predictor on data held-out from training, preprocessing (such as standardization, feature selection, etc.) and similar data transformations similarly should be learnt from a training set and applied to held-out data for prediction:
>>>
>>> from sklearn import preprocessing
>>> X_train, X_test, y_train, y_test = train_test_split(
...     iris.data, iris.target, test_size=0.4, random_state=0)
>>> scaler = preprocessing.StandardScaler().fit(X_train)
>>> X_train_transformed = scaler.transform(X_train)
>>> clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
>>> X_test_transformed = scaler.transform(X_test)
>>> clf.score(X_test_transformed, y_test)

A Pipeline makes it easier to compose estimators, providing this behavior under cross-validation:
>>>
>>> from sklearn.pipeline import make_pipeline
>>> clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
>>> cross_val_score(clf, iris.data, iris.target, cv=cv)



------
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

# draw visualization of parameter effects
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()
'''
