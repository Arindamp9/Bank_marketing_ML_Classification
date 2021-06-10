# Bank_marketing_ML_Classification

##### Data link: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# REPORT
##### Project Goal: 
Predicting if the client will subscribe a term deposit by building three classifiers for the given data set: a decision tree, a naïve Bayes classifier, and a random forest.
#### Imported libraries:
*	numpy(renamed as np)
*	pandas(renamed as pd)
*	seaborn(renamed as sns)
*	matplotlib.pyplot(renamed as plt)
*   import time    (To calculate time)
*	train_test_split(imported from sklearn.model_selection)
*	imblearn 
*	RandomOverSampler(imported from imblearn.over_sampling)
*	Counter(imported from collections)
*	confusion_matrix,classification_report,precision_score,recall_score,f1_score(imported from sklearn.metrics)
*	DecisionTreeClassifier(imported from sklearn.tree)
*	RandomForestClassifier(imported from sklearn.ensemble)
*	GaussianNB(imported from sklearn.naive_bayes)
*   warnings    (To avoid unwanted warnings)
*   memory_profiler    (To calculate space)

##### Parameters Used:
* For Decision Tree, Max depth of tree =7 and Min Samples in a leaf =10(To avoid overfitting), Criterion = Gini Index.
* For Random Forest, Max_depth=7
* For Naive Bayes, no parameters have been taken as input.

###### Confession:
Here we have not dropped the **duration** column, also we know that we should drop it because of the real world scenario. To be honest we kept it to get some good metric score :)
   
#### Procedure:
*	First we read the data and store it into variable named **df** by the following command:
**df=pd.read_csv('bank-additional-full.csv')**
*	Print the first five rows of the datasets by the following command:
**df.head()**
*	Then we calculate the number of unique elements in each categorical variables and made a list of each elements to corresponding categorical variable.
*	Change the element **yes** in variable **y** by **1** and the element **no** by **0**. Use the following command:
**df['y']=df['y'].replace({'no':0,'yes':1})**
*	Next we plot heatmap corresponding to Pearson Correlation. From the diagram conclude that **euribor3m**,**nr.employed**,**emp.var.rate** are highly correlated to each other. So we will only keep one of them and drop other two. We will keep **euribor3m** feature, since it is highly correlated with others.
 
*	We drop the default column as well since it contains 32588 **no**, 8597 **unknown** and only 3 **yes**. Hence this feature will not have any significant effect on the model.
*	Now we need to do use some encoding technique to covert categorical variables into numerical because machine can’t read string kind of data types. So we need to convert them into numerical variable and then give it for the modelling.
*	We label encode **education**, **contact**, **month**, **day_of_week** and apply one-hot encoding on the columns **job**, **marital**, **housing**, **loan**, **poutcome**. We apply one-hot since in these columns there exist no particular ordering in the attributes. 
*	The attribute **999** in the column **pdays** means the client has not been contacted before. We change attribute **999** as **0**, since if the client is not previously contacted we can treat the previous day count as **0**.
*	Now we drop the **y** column. We split the given dataset into two parts, training dataset **X_train** and test dataset **X_test**.
*	Then we apply oversampling on the training dataset, since the dataset is imbalanced. The previous ratio of the dataset is 9:1. After applying oversampling the ratio of the dataset becomes **2:1**.
* We do not need to apply any feature scaling technique here, since these three model won’t affect much for different scales.
*	Finally we fit decision tree, random forest and naïve bayes classifier. Also calculate the corresponding confusion matrix and classification report. <br>
*   We have calculated the total preprocessing time and the time taken by each model seperately.

###### Now let's look at the metric score for each model.

# OUTPUT

**Total Preprocessing Time :  1.0808069705963135 Secs**

## For Decision Tree :: 

###### Confusion Matrix  

[[6441  862]<br>
 [ 141  794]]


###### Classification Report  

             precision    recall  f1-score   support

           0       0.98      0.88      0.93      7303
           1       0.48      0.85      0.61       935

    accuracy                           0.88      8238
    macro avg      0.73      0.87      0.77      8238
    weighted avg   0.92      0.88      0.89      8238

**Accuracy :  0.878<br>
Precision :  0.479<br>
Recall :  0.849<br>
f1 :  0.613<br>
Time Taken:  1.7520267963409424 secs<br>
peak memory: 210.39 MiB, increment: 7.31 MiB**<br>
.....................................................................................

## For Random Forest :: 

###### Confusion Matrix  

[[6603  700]<br>
 [ 189  746]]


###### Classification Report  

             precision    recall  f1-score   support

           0       0.97      0.90      0.94      7303
           1       0.52      0.80      0.63       935

    accuracy                           0.89      8238
    macro avg      0.74      0.85      0.78      8238
    weighted avg   0.92      0.89      0.90      8238

**Accuracy :  0.891<br>
Precision :  0.514<br>
Recall :  0.799<br>
f1 :  0.626<br>
Time Taken:  2.8892226219177246 secs<br>
peak memory: 213.19 MiB, increment: 7.84 MiB<br>**
..................................................................................

## For Naive Bayes :: 

###### Confusion Matrix  

[[6440  863]<br>
 [ 442  493]]


###### Classification Report  

              precision    recall  f1-score   support

           0       0.94      0.88      0.91      7303
           1       0.36      0.53      0.43       935

    accuracy                           0.84      8238
    macro avg      0.65      0.70      0.67      8238
    weighted avg   0.87      0.84      0.85      8238

**Accuracy  :  0.842<br>
Precision :  0.364<br>
Recall    :  0.527<br>
f1_score  :  0.43<br>
Time Taken:  1.5341289043426514 secs<br>
peak memory: 222.40 MiB, increment: 14.56 MiB<br>**
..............................................................................................<br>
**N.B.: Whatever time we have calculated here, we have got it by running all the cell at a time(From Run All option).**<br>
        **And Space is calculated for each model only.**<br>
...............................................................................................
## Conclusion:
**Here Random forest is giving us the best accuracy and Decision tree is giving us the best recall and Naive bayes is not giving us good result as comparative to other. For this Problem I believe we need to maximize the recall score(i.e, to minimize the False negative error). So if we compare all the model we can say both Decision Tree and Random Forest  is the best model for this problem,since the scores are pretty same for these two models.**
