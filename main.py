# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:15:08 2020

@author: Guru
"""
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np

st.title("Streamlit and Machine Learning")
st.write("""
         # Explore Different Classifiers
         Which one is the best?
         """
    )


dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

classifier_name = st.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    
    X = data.data
    y = data.target
    return X,y
    
        
X, y = get_dataset(dataset_name)
st.write("Shape of data is ", X.shape)
st.write("Number of Classes is", len(np.unique(y)))   

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        MaxDepth = st.sidebar.slider("Max Depth", 2, 15)
        NEstimators = st.sidebar.slider("Number of Estimators", 1, 100)
        params["MaxDepth"] = MaxDepth
        params["NEstimators"] = NEstimators
    return params
        
        
P = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors = P["K"], )
        
        elif clf_name == "SVM":
            
            clf = SVC(C=P["C"])
        
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators = P["NEstimators"], max_depth=P["MaxDepth"],
                                         )
            
        return clf
        
clf = get_classifier(classifier_name, P)


st.write("""
         # Classification
         """
    )        
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
acc = accuracy_score(y_test, y_predict)

        
st.write(f"Classifer = {classifier_name}")
st.write(f"Classifer Accuracy= {acc}")        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
