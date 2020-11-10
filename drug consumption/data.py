# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 2020

Data obtained from: http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
@author: Ethan Bosworth

A piece of work for university originally written in MATLAB but converted to Python

"""
#%% import modules

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score


#set plotting style
plt.style.use("seaborn")
# create colourblind friendly colours for use throughout
user_colour = "#377eb8"
non_colour = "#ff7f00"
palette = ["#377eb8","#ff7f00"]

#%% import data

#the data contains many different variables however the task asked for just that based on 
#Heroin and the 7 psychology traits 
#so create a columns list of only the important columns to import
columns = [6,7,8,9,10,11,12,23]

#import the data from the data file
data = pd.read_csv("Data/drug_consumption.data",header = None,usecols = columns)

#give the columns correct names
data.columns = ["Nscore","Escore","Oscore","Ascore","Cscore","Impulsive","SS","Heroin_use"]

del columns
#%% Creating the target variable
# the task calles for the target variable to be if a person used Heroin in the past year
# This means a user is any of CL6,CL5,CL4,CL3

data["Heroin_use"].replace(["CL6","CL5","CL4","CL3"],1,inplace = True)
data["Heroin_use"].replace(["CL2","CL1","CL0"],0,inplace = True)

#%% Descriptive statistics
#finding the mean value and standard deviation of each of the 7 charecteristics 
#for users and non-users with 95% confidence intervals

#create a dataframe for users and non users
users = data[data["Heroin_use"] == 1].drop("Heroin_use",axis = 1)
non_users = data[data["Heroin_use"] == 0].drop("Heroin_use",axis = 1)

#create a function to get the mean, standard deviation and 95% confidence interval
def mean_std_ci(data):
    data_return = pd.DataFrame(data.mean()) # find the mean of each column
    data_return[1] = np.std(data)# find the standard deviation of each column
    # find the upper and lower values of the 95% confidence interval
    data_return[2] = data_return[0]+1.960*(data_return[1]/np.sqrt(len(data)))
    data_return[3] = data_return[0]-1.960*(data_return[1]/np.sqrt(len(data)))
    data_return.columns = ["Mean","Std","95% upper","95% lower"]
    return data_return

# returns the statistics for each of the groups Users and Non-Users
users_stats = mean_std_ci(users)
non_users_stats = mean_std_ci(non_users)
print("Users")
print(users_stats)
print("\n Non-Users")
print(non_users_stats)

#creates a plot for the statistics between the classes
u = sns.pointplot(data = users,color = user_colour)
n = sns.pointplot(data = non_users,color = non_colour)
u.set_title('Psychological scores of Users and Non-Users of Heroin')
leg = u.legend(labels = ["User","Non-User"])
leg.legendHandles[0].set_color(user_colour)
leg.legendHandles[1].set_color(non_colour)
plt.show()

del u,n,leg
#%%  Significance evaluation

#from scipy the t-test will give a p value for each variable
# and will take a p value 0f 0.05
P_significant = 0.05

#preform a t-test
p_values = pd.DataFrame(stats.ttest_ind(users, non_users,equal_var=False).pvalue)
p_values.columns = ["p_value"]
p_values["Significant"] = p_values["p_value"] < P_significant

print(p_values)
#all are significant with Escore being the closest to being not significant

del P_significant
#%% Creating single variable predictors

#%%% Setting up functions

#create a function to get the data in a usable form based on the variable
def melt_data(Variable):
    #create two temporary variables for each variable users and non-users and convert to percentages
    Score_U = pd.DataFrame(users[Variable].value_counts().sort_index())
    Score_U = (Score_U/Score_U.sum())*100 #convert to percentage
    Score_N = pd.DataFrame(non_users[Variable].value_counts().sort_index())
    Score_N = (Score_N/Score_N.sum())*100
    
    Score = pd.concat([Score_U,Score_N],axis = 1) # merge two sets of data
    Score.index =np.round(Score.index,2) # round the values of the scores to make it easier to read
    Score.columns = ["Users","Non Users"] # set column names
    Score.fillna(0,inplace = True) # fill missing data with 0 as no people are in that point
    # melt the data to have users and non-users in the same dataframe with a variable "users" causing distinction between them
    Score_melt = Score.reset_index().melt(id_vars="index", var_name="User") 
    return Score_melt

#create a function for plotting a graph of users vs non-users for the variable
def plot_variable(data,variable):
    plt.figure(figsize=(12,4))
    a = sns.barplot(x = "index",y = "value",data =  data,hue = "User",palette = palette)
    a.set_title(variable)
    a.set_xticklabels(a.get_xticklabels(), rotation=80, ha="right")
    a.set_ylabel("Percentage")
    plt.show()

#create a function to give the best single value predictor for a variable
def best_prediction(melt_data,variable):
    #create a reverse variable and is true if mean of users is less than mean of non-users
    reverse_val = [users[variable].mean(),non_users[variable].mean()]
    reverse = reverse_val[0] < reverse_val[1]

    #takes all the unique values of score
    Predict = pd.DataFrame(melt_data["index"].unique())
    Predict.columns = ["index"]
    #sets the user/non-user variable  to the percentage for each value of score from the melt data
    Predict["User"] = pd.DataFrame(melt_data[melt_data["User"] == "Users"].value).set_index(Predict.index)
    Predict["Non-User"] = pd.DataFrame(melt_data[melt_data["User"] == "Non Users"].value).set_index(Predict.index)
    #creates some empty variables for now which will be filled in by the respective values
    #TP (True Positive) is anybody identified as a User correctly
    #FP (False Positive) is a Non-User identified as a User
    #TN (True Negative) is anybody identified as a Non-user correctly
    #FN (False Negative) is a User identified as a Non-User
    Predict["TP"],Predict["FP"],Predict["TN"],Predict["FN"],Predict["Error"] = [Predict["User"],Predict["User"],Predict["User"],Predict["User"],Predict["User"]]
    
    for i in Predict.index:
        if reverse == False: #check if reverse is false not
            #Predict anybody over i is a user
            
            Predict["TP"][i] = Predict[Predict["index"] > Predict["index"][i]]["User"].sum()
            Predict["FP"][i] = Predict[Predict["index"] > Predict["index"][i]]["Non-User"].sum()
            Predict["TN"][i] = Predict[Predict["index"] <= Predict["index"][i]]["Non-User"].sum()
            Predict["FN"][i] = Predict[Predict["index"] <= Predict["index"][i]]["User"].sum()
            #error is the sum of FP and FN over the sum of all
            Predict["Error"][i] = (Predict["FP"][i]+Predict["FN"][i])/sum([Predict["TN"][i],Predict["FN"][i],Predict["FP"][i],Predict["TP"][i]])
        else: #reverse is true
            #Predict anybody under i is a user
            Predict["TP"][i] = Predict[Predict["index"] < Predict["index"][i]]["User"].sum()
            Predict["FP"][i] = Predict[Predict["index"] < Predict["index"][i]]["Non-User"].sum()
            Predict["TN"][i] = Predict[Predict["index"] >= Predict["index"][i]]["Non-User"].sum()
            Predict["FN"][i] = Predict[Predict["index"] >= Predict["index"][i]]["User"].sum()
            Predict["Error"][i] = (Predict["FP"][i]+Predict["FN"][i])/sum([Predict["TN"][i],Predict["FN"][i],Predict["FP"][i],Predict["TP"][i]])
            
    #find the part of the dataframe with minimum error and return the score,error and if it was reversed or not
    best = pd.DataFrame([(Predict[Predict["Error"] == Predict["Error"].min()]["index"]).values[0],Predict["Error"].min(),reverse])
    best.index = ["Score","Error","reversed"]
    best.columns = [variable]
    return best
#%%% running over the variables

#create a score prediction variable to hold all the results
Score_pred = pd.DataFrame(["Score","Error","reversed"])
Score_pred.columns = ["index"]
Score_pred = Score_pred.set_index("index")

for i in users.columns: # for each variable create a graph and add to the Score prediction variable
    Score = melt_data(i)
    plot_variable(Score,i)
    Score_pred = pd.concat([Score_pred,best_prediction(Score,i)],axis = 1)
    
print(Score_pred) # as it turns out SS is the best predictor
#creates an accuracy value for each predictor
Accuracy = pd.DataFrame(1-Score_pred.loc["Error"]).T
Accuracy.index = ["Accuracy"]
Accuracy = Accuracy.astype("float")
del Score

#%% 1NN and 3NN 
#task asks to create a 1NN and a 3NN predictor

#create X and y variables from the data
X = data.drop("Heroin_use",axis = 1)
y = data["Heroin_use"]
#check the class balance
print(y.value_counts()) # returns a heavy class imbalance which needs to be fixed first
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1) # split into train and test data

#random oversampler will oversample the minority class in the data to balance the classes
ros = RandomOverSampler(random_state=1,sampling_strategy="auto")

X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
print(y_resampled.value_counts()) # check the class ratio of the resampled data

#create a 1NN classifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_resampled,y_resampled)
y_pred = neigh.predict(X_test) # predict
acc = accuracy_score(y_test,y_pred) # print the accuracy of prediction
print(acc)

Accuracy["1NN"] = acc # add the accuracy to the list of accuracies

#repeat for 3NN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_resampled,y_resampled)
y_pred = neigh.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(acc)

Accuracy["3NN"] = acc

del ros,neigh

#%% Fishers Linear Discriminant

#create a function to find fishers linear discriminant for 2 classes
def fisher(X,y,Weight):
    Xy = pd.concat([X,y],axis = 1) # make sure all the data is together
    #split into users and non users and remove the target variable
    users = Xy[Xy["Heroin_use"] == 1].drop("Heroin_use",axis = 1) 
    non_users = Xy[Xy["Heroin_use"] == 0].drop("Heroin_use",axis = 1)
    #find the mean of both classes
    mean_users = users.mean(axis=0)
    mean_non_users = non_users.mean(axis = 0)
    # create a covariance matrix for each class and sum them
    Sw = np.cov(users.T) + np.cov(non_users.T)
    inv_S = np.linalg.inv(Sw) #invert Sw
    # use the equation inv_Sw * (mean_a - mean_b) to find the vector of the discriminant
    res = pd.DataFrame(inv_S.dot(mean_users - mean_non_users)).T

    #apply the discriminant to each class and find the mean on the line
    mu1 = (np.mean(np.dot(res, users.T)))*Weight # multiply by a weight in an attempt to make up for imbalanced classes
    mu2 = np.mean(np.dot(res, non_users.T))
    #find the mean of the two to create a threshold between
    Threshold = (mu1+mu2)/2
    return res,Threshold

range_weights = np.arange(0.1,5,0.1) #create a range of weights
acc = pd.DataFrame(range_weights) #convert to dataframe
#create colmumns ready to take the overall accuracy and accuracy for each class
acc.columns = ["Weight"] 
acc["Accuracy"] = acc["Weight"]
acc["Acc_user"] = acc["Weight"]
acc["Acc_non_user"] = acc["Weight"]

#loop over the weights to test
for i in acc.index:
    res,Threshold = fisher(X_train,y_train,acc["Weight"][i]) #run the function with the weight
    X = X_test.copy() # take the test data
    Output = pd.DataFrame(np.dot(res, X.T)).T #apply discriminant to the test data
    Output = Output.set_index(X.index)
    #if the output is over the threshold classify as a user
    Output["prediction"] = (Output >= Threshold).replace([True,False],[1,0])
    Output["User"] = y_test #import actual class
    Output["Correct"] = Output["User"] == Output["prediction"] #check if correct or not
    #output total accuracy, user accuracy and non user accuracy to the dataframe
    acc["Accuracy"][i] = Output["Correct"].mean() 
    acc["Acc_user"][i] = Output[Output["User"] == 1].mean().Correct
    acc["Acc_non_user"][i] = Output[Output["User"] == 0].mean().Correct
    
#create a plot to see the best values of the weight
a = sns.lineplot(data = acc,x = "Weight",y = "Accuracy" )
a = sns.lineplot(data = acc,x = "Weight",y = "Acc_user" )
a = sns.lineplot(data = acc,x = "Weight",y = "Acc_non_user" )
a.legend(labels = ["Accuracy","User","Non-User"])

plt.show()
#as is seen from the plot there s no best answer. a higher accuracy gives a lower user accuracy
#the maximum accuracy is over 90% however to get a high accuracy for users a lower overall accuracy should be taken

      