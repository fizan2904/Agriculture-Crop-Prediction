import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

plt.style('ggplot')

def linear_mode():
	dataTrain = pd.read_csv("/Users/senora/Desktop/train.csv")
	dataTest = pd.read_csv("/Users/senora/Desktop/test.csv")

	x_train = dataTrain[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', 'AREA UNDER CULTIVATION']]
	y_train = dataTrain[['YIELD']]

	x_test = dataTest[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', 'AREA UNDER CULTIVATION']]
	y_test = dataTest[['YIELD']]

	ols = linear_model.LinearRegression()
	model = ols.fit(x_train, y_train)
	accuracy = ols.score(x_test, y_test)

	return ols, model

def svm_mode():
	dataTrain = pd.read_csv("/Users/senora/Desktop/Keshu-agriculture/app/old_data/arj_formatted.csv")

	X = dataTrain[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', 'AREA UNDER CULTIVATION PADDY']]
	y = dataTrain[['YIELD PADDY']]

	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	y_rbf = svr_rbf.fit(X, y.values.ravel()).predict(X)
	y_lin = svr_lin.fit(X, y.values.ravel()).predict(X)
	y_poly = svr_poly.fit(X, y.values.ravel()).predict(X)

	lw = 2
	plt.scatter(X, y, color='darkorange', label='data')
	plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
	plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
	plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
	plt.xlabel('data')
	plt.ylabel('target')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

svm_mode()