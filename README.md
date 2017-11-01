# Agriculture Crop Prediction

An api to predict what kind of crop is better to actually grow that crop to gain more profit based on the previous data

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* Python 3
* Flask
* Sklearn
* numpy
* pandas

### Installing

A step by step series of examples that tell you have to get a development env running

Install Flask

```
sudo pip3 install flask
```

Install Sklearn

```
sudo pip3 install sklearn
```

Install Numpy

```
sudo pip3 install numpy
```

Install Pandas

```
sudo pip3 install pandas
```

## Deployment

To run main server

```
cd /path/to/project
python3 main.py
```

## Accessing API

All the api can be accessed by adding the following url to base url http://localhost:5000/predict/

*kernel = { gamma, linear, poly, linearwithc, polywithdegree, rbc}<br />
*crop = { paddy, maize, cereals}<br />
*reg_type = { linear, logistic}<br />
*river = { arjuna, koushika }

### Linear Regression

```
/kernel_regression/<reg_type>/<river>/<crop>
```

### SVM

```
/svm/<reg_type>/<river>/<crop>
```

### KNN

```
/kernel_regression/<river>/<crop>
```
