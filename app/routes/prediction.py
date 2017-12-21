import json
import numpy as np
from app import app
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import _pickle as pickle
from os import path

style.use('ggplot')

basepath = path.dirname(__file__)
path0 = path.abspath(path.join(basepath, "..", "old_data", "arj_formatted.csv"))
path1 = path.abspath(path.join(basepath, "..", "old_data", "train.csv"))
path2 = path.abspath(path.join(basepath, "..", "old_data", "test.csv"))
path3 = path.abspath(path.join(basepath, "..", "old_data", "train_koushika.csv"))
path4 = path.abspath(path.join(basepath, "..", "old_data", "test_koushika.csv"))
data = pd.read_csv(path0)
print(path1)
dataTrain_arjuna = pd.read_csv(path1)
dataTest_arjuna = pd.read_csv(path2)

dataTrain_koushika = pd.read_csv(path3)
dataTest_koushika = pd.read_csv(path4)

@app.route('/predict/kernel_regression/<type_of_reg>/<river>/<crop>')
def predict(river, crop, type_of_reg):
    if river == "arjuna":
        dataTrain = dataTrain_arjuna
        dataTest = dataTest_arjuna
    elif river == "koushika":
        dataTrain = dataTrain_koushika
        dataTest = dataTest_koushika
    else:
        return json.dumps({'error':'Not valid river name'}), 500, {'ContentType':'application/json'}
    
    if crop == "paddy":
        area_name = "AREA UNDER CULTIVATION PADDY"
        yield_name = "YIELD PADDY"
    elif crop == "maize":
        area_name = "AREA UNDER CULTIVATION MAIZE"
        yield_name = "YIELD MAIZE"
    elif crop == "cereals":
        area_name = "AREA UNDER CULTIVATION CEREALS"
        yield_name = "YIELD CEREALS"
    else:
        return json.dumps({'error':'Not valid crop name'}), 500, {'ContentType':'application/json'}
    
    x_train = dataTrain[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', area_name]]
    y_train = dataTrain[[yield_name]]
    x_test = dataTest[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', area_name]]
    y_test = dataTest[[yield_name]]
    if type_of_reg == "linear":
        ols = linear_model.LinearRegression()
    elif type_of_reg == "logistic":
        ols = linear_model.LogisticRegression(C=1e5)
    else:
        return json.dumps({'error': 'Wrong regression module'}), 500, {'ContentType': 'application/json'}
    model = ols.fit(x_train, y_train)
    accuracy = ols.score(x_test, y_test)
    data = {
	    'prediction': np.array(model.predict(x_test)[0:5]).tolist(),
	    'years': np.array([2006, 2007, 2008, 2009, 2010]).tolist()
    }
    return json.dumps({'accuracy':accuracy, 'data':data}), 200, {'ContentType':'application/json'}

@app.route('/predict/svm/<kernel>/<river>/<crop>')
def predict_svm(river, crop, kernel):
    if river == "arjuna":
        dataTrain = dataTrain_arjuna
        dataTest = dataTest_arjuna
    elif river == "koushika":
        dataTrain = dataTrain_koushika
        dataTest = dataTest_koushika
    else:
        return json.dumps({'error':'Not valid river name'}), 500, {'ContentType':'application/json'}
    
    if crop == "paddy":
        area_name = "AREA UNDER CULTIVATION PADDY"
        yield_name = "YIELD PADDY"
    elif crop == "maize":
        area_name = "AREA UNDER CULTIVATION MAIZE"
        yield_name = "YIELD MAIZE"
    elif crop == "cereals":
        area_name = "AREA UNDER CULTIVATION CEREALS"
        yield_name = "YIELD CEREALS"
    else:
        return json.dumps({'error':'Not valid crop name'}), 500, {'ContentType':'application/json'}
    
    x_train = dataTrain[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', area_name]]
    y_train = dataTrain[[yield_name]]
    x_test = dataTest[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', area_name]]
    y_test = dataTest[[yield_name]]
    if kernel == "gamma":
        ols = svm.SVC(gamma=0.001, C=100.)
    elif kernel == "linear":
        ols = svm.SVC(kernel='linear')
    elif kernel == "poly":
        ols = svm.SVC(kernel='poly')
    elif kernel == "linearwithc":
        ols = svm.SVC(kernel='linear', C=1e3)
    elif kernel == "polywithdegree":
        ols = svm.SVC(kernel='poly', C=1e3, degree=2)
    elif kernel == "rbc":
        ols = svm.SVC(kernel='rbf')
    else:
        return json.dumps({'error': 'Wrong kernel type'}), 500, {'ContentType': 'application/json'}

    model = ols.fit(x_train, y_train)
    accuracy = ols.score(x_test, y_test)
    data = {
	    'prediction': np.array(model.predict(x_test)[0:5]).tolist(),
	    'years': np.array([2006, 2007, 2008, 2009, 2010]).tolist()
    }
    return json.dumps({'model':str(model), 'data':data}), 200, {'ContentType':'application/json'}

@app.route('/predict/knn/<river>/<crop>')
def predict_knn(river, crop):
    if river == "arjuna":
        dataTrain = dataTrain_arjuna
        dataTest = dataTest_arjuna
    elif river == "koushika":
        dataTrain = dataTrain_koushika
        dataTest = dataTest_koushika
    else:
        return json.dumps({'error':'Not valid river name'}), 500, {'ContentType':'application/json'}
    
    if crop == "paddy":
        area_name = "AREA UNDER CULTIVATION PADDY"
        yield_name = "YIELD PADDY"
    elif crop == "maize":
        area_name = "AREA UNDER CULTIVATION MAIZE"
        yield_name = "YIELD MAIZE"
    elif crop == "cereals":
        area_name = "AREA UNDER CULTIVATION CEREALS"
        yield_name = "YIELD CEREALS"
    else:
        return json.dumps({'error':'Not valid crop name'}), 500, {'ContentType':'application/json'}
    
    x_train = dataTrain[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', area_name]]
    y_train = dataTrain[[yield_name]]
    x_test = dataTest[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', area_name]]
    y_test = dataTest[[yield_name]]
    ols = KNeighborsClassifier()
    model = ols.fit(x_train, y_train)
    fi = pickle.dumps(model)
    accuracy = ols.score(x_test, y_test)
    data = {
	    'prediction': np.array(model.predict(x_test)[0:5]).tolist(),
	    'years': np.array([2006, 2007, 2008, 2009, 2010]).tolist()
    }
    return json.dumps({'model':str(model), 'data':data}), 200, {'ContentType':'application/json'}