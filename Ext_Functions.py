
import pandas as pd
import numpy as np

def str_to_bool_series(s_orig):
    s = s_orig.copy()
    for index, value in s.items():
        if(value=='Yes'):
            s[index] = 1
        elif(value=='No'):
            s[index] = 0
        elif(value=='Male'):
            s[index] = 1
        elif(value == 'Female'):
            s[index] = 0
        elif(value=='Positive'):
            s[index] = 1
        elif (value == 'Negative'):
            s[index] = 0
#         elif(value==1):
#             s[index] = True
#         elif(value==0):
#             s[index] = False
    return s


def nan2num(dataframe):
    """
    :param dataframe: Pandas series of features
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    # dataframe = pd.DataFrame(dataframe).drop(extra_feature, 1)
    c_cdf = {}
    c_cdf = dataframe
    for column in dataframe.columns:
        if column == 'Age':
            avg_age = pd.Series.mean(dataframe[column])
            c_cdf[column].replace(to_replace= np.nan , value=50, inplace = True)
        else:
            replacing_value = np.random.choice(dataframe[column])
            c_cdf[column].replace(to_replace= np.nan , value=replacing_value, inplace = True)
    return pd.DataFrame(c_cdf)


def print_dist_features(df, x_train, x_test):
    print("Feature            | Train % | Test % | Difference")
    for column in df:
        if(column=='Age'):
            age_mean_train = round(x_train[column].mean(),2)
            print(column ,end =  " "*(21-len(column))) 
            print(age_mean_train, end = " "*(5))
            age_mean_test = round(x_test[column].mean(),2)
            print(age_mean_test, end = " "*(5))
            difference = round(age_mean_train-age_mean_test,2)
            print(difference, end = " "*(5))
            print()
        elif(column=='Family History'):
            train_count = round(x_train[column].value_counts(ascending = True)/len(x_train[column])*100,2)
            test_count = round(x_test[column].value_counts(ascending = True)/len(x_test[column])*100,2)
            difference = round(train_count[1]-test_count[1],2)
            print(column, end=" "*(21-len(column)))
            print(train_count[1], end = " "*(10-len(str(train_count[1]))))
            print(test_count[1], end = " "*(10-len(str(test_count[1]))))
            print(difference)
        else:
            train_count = round(x_train[column].value_counts(ascending = True)/len(x_train[column])*100,2)
            test_count = round(x_test[column].value_counts(ascending = True)/len(x_test[column])*100,2)
            difference = round(train_count[1]-test_count[1],2)
            print(column, end=" "*(21-len(column)))
            print(train_count[1], end = " "*(10-len(str(train_count[1]))))
            print(test_count[1], end = " "*(10-len(str(test_count[1]))))
            print(difference)
    return

import seaborn as sbn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
            
def feature_corr(t1d_df):
    fig, axes = plt.subplots(4, 5,figsize=(20, 15) )
    sbn.set_context("paper", font_scale=0.7)
    sbn.set_theme(style="darkgrid", palette="Set1")
    fig.suptitle("Relationship between features and label")
    plt.rcParams.update({'font.size': 22})
    i=0
    for column in t1d_df:
        if(column=='Gender'):
            feat_lab = sbn.countplot(ax = axes[i//5,i%5], x='Gender', hue = 'Diagnosis', data = t1d_df)
            feat_lab.set(xticklabels=['Male', 'Female'])
        elif(column=='Age'):
            i-=1
#             feat_lab = sbn.countplot(ax = axes[i//5,i%5], x='Age', hue = 'Diagnosis', data = t1d_df)
#             feat_lab.xaxis.set_major_locator(ticker.LinearLocator(10))
        elif(i>=1):
            feat_lab = sbn.countplot(ax = axes[i//5,i%5], x=column, hue = 'Diagnosis', data = t1d_df)
            feat_lab.set(xticklabels=['No', 'Yes'])
        i+=1
    return()


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
def cv_kfold(X, y, C, penalty, K):
    """   
    :param X: Training set samples
    :param y: Training set labels 
    :param C: A list of regularization parameters
    :param penalty: A list of types of norm
    :param K: Number of folds
    :return: A dictionary as explained in the notebook
    """
    kf = StratifiedKFold(n_splits=K)
    validation_dict = []
    for c in C:
        for p in penalty:
            logreg = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, multi_class='ovr')
            loss_val_vec = np.zeros(K)
            auc_vec = np.zeros(K)
            score_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx],y[val_idx]
                y_pred,_=pred_log(logreg,x_train,y_train,x_val,flag = True)
                loss_val_vec[k]=log_loss(y_val,y_pred)
                auc_vec[k] = roc_auc_score(y_val, y_pred[:, 1])
                score_vec[k] = logreg.score(x_val, y_val)
                k=k+1
            validation_dict.append({'C':c,'penalty':p,'Mu':np.mean(loss_val_vec),'Sigma':np.std(loss_val_vec), 
                                   'AUC':auc_vec.max(), 'Score':score_vec.max()})
    return validation_dict


def pred_log(logreg, X_train, y_train, X_test, flag=False):
    """
    :param logreg: An object of the class LogisticRegression
    :param X_train: Training set samples
    :param y_train: Training set labels 
    :param X_test: Testing set samples
    :param flag: A boolean determining whether to return the predicted he probabilities of the classes (relevant after Q11)
    :return: A two elements tuple containing the predictions and the weightning matrix
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    logreg.fit(X_train,y_train)
    if flag==True:
        y_pred_log = logreg.predict_proba(X_test)
    else:
        y_pred_log=logreg.predict(X_test)
    w_log=logreg.coef_
    # -------------------------------------------------------------------------
    return y_pred_log, w_log

def get_logreg_score(model, X_train_gs, X_test_gs, y_train_gs, y_test_gs):
    model.fit(X_train_gs, y_train_gs)
    scores = []
    y_pred = model.predict(X_test_gs)    
    scores.append(model.score(X_test_gs, y_test_gs))
    scores.append(roc_auc_score(y_test_gs,y_pred))
    scores.append(log_loss(y_test,y_pred))
    return


from sklearn import metrics
# function that prints different parameters to evaluate the model 
def parameter_evaluation(y_test,y_train,y_test_pred,y_train_pred):
    print('Test set results:')
    print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_test_pred))) + "%")
#     print("Precision is: " + str("{0:.2f}".format(100 * metrics.precision_score(y_test, y_test_pred))) + "%")
#     print("recall is: " + str("{0:.2f}".format(100 * metrics.recall_score(y_test, y_test_pred))) + "%")
    print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_test_pred, average='macro'))) + "%")
    print("AUC is: " + str("{0:.2f}".format(100 * metrics.roc_auc_score(y_test,y_test_pred))) + "%"+'\n')
    print('Train set results:')
    print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_train, y_train_pred))) + "%")
#     print("Train precision is: " + str("{0:.2f}".format(100 * metrics.precision_score(y_train, y_train_pred))) + "%")
#     print("Train recall is: " + str("{0:.2f}".format(100 * metrics.recall_score(y_train, y_train_pred))) + "%")
    print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_train, y_train_pred, average='macro'))) + "%")
    print("AUC is: " + str("{0:.2f}".format(100 * metrics.roc_auc_score(y_train,y_train_pred))) + "%")
    return