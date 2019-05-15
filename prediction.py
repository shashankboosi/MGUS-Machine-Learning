
"""

Code for prediction of deaths using Monoclonal Gammopathy(MGUS)

7 different Machine Learning algorithms on the data-set for comparison.
1) Support Vector Machines(SVM)
2) Linear Discriminant Analysis
3) Logistic Regression
4) K Nearest Neighbors
5) Naive Bayes
6) Decision Tree Classifier
7) Random Forest Classifier

SVM is performing better than the remaining algorithms.
"""

# Imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# SKLearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Ignoringthe warnings
warnings.filterwarnings('ignore')

# Calling the data from using panda and verifying
main_file=pd.read_csv('./Dataset/train_gammo.csv')
main_file = main_file.drop('id', axis=1)
# del main_file["ptime"]
# del main_file["id"]
print(main_file.corr())
print(main_file.describe())
print(main_file.groupby("death").size())
print(main_file.groupby("sex").size())

#T he missing values from the haemoglobin column have been replaced
haemoglobin=main_file["hgb"].mean()
print(haemoglobin)
main_file["hgb"] = main_file["hgb"].fillna(haemoglobin)
age = main_file["age"].mean()
print(age)
# The missing values from the creatinine column have been replaced
creatinine=main_file["creat"].mean()
print(creatinine)
main_file["creat"] = main_file["creat"].fillna(creatinine)

# The missing values from the serum spike column have been replaced
serum_spike=main_file["mspike"].mean()
print(serum_spike)
main_file["mspike"] = main_file["mspike"].fillna(serum_spike)

#Checking the correlation between two groups
#print(main_file[['pstat', 'death']].groupby(['pstat'], as_index=False).mean().sort_values(by='death', ascending=False))
## print(main_file[['futime', 'ptime']].groupby(['futime'], as_index=False).mean().sort_values(by='ptime', ascending=False))
#print(main_file[['pstat', 'ptime']].groupby(['pstat'], as_index=False).mean().sort_values(by='ptime', ascending=False))
#print(main_file[['death', 'futime']].groupby(['death'], as_index=False).mean().sort_values(by='futime', ascending=False))
#print(main_file.groupby("mspike").size())

sns.set(style="ticks")
sns.pairplot(main_file, hue="death")
plt.show()

# Transforming the attribute sex from male and female into numeric
le = LabelEncoder()
main_file['sex'] = le.fit_transform(main_file['sex'].map({'F': 0, 'M': 1}).astype(str))
print(main_file.head())

print(main_file[['sex', 'death']].groupby(['sex'], as_index=False).mean().sort_values(by='death', ascending=False))

# corr=main_file.corr()#["death"]
# plt.figure(figsize=(10, 10))
#
# sns.heatmap(corr, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='YlGnBu',linecolor="white")
# plt.title('Correlation between features')
# plt.show()

# Splitting the training and the testing data in the ratio of 1:4
X = main_file.ix[:, 0:8]
Y = main_file["death"]
seed = 10
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2,random_state=seed)
print(len(X_train))
print(len(X_test))

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('The head is :')
print('/n')
print(X_train.shape)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_result = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    names.append(name)
    results.append(cv_result)

for i in range(len(names)):
    print(names[i],results[i].mean())

ax = sns.boxplot(data=results)
ax.set_xticklabels(names)
plt.show()


# Support Vector Machines
for kernel in ('linear', 'poly', 'rbf'):
    svm = SVC(kernel=kernel)
    svm.fit(X_train,Y_train)
    svm_score = round(svm.score(X_train, Y_train) * 100, 2)
    print('SVM Score: \n', svm_score)

    predictions_svm = svm.predict(X_test)

    # Accuracy Score of the svm
    print("Accuracy Score of svm is:")
    print(accuracy_score(Y_test, predictions_svm))

    print("Classification Report of svm is:")
    print(classification_report(Y_test, predictions_svm))
    con = confusion_matrix(Y_test, predictions_svm)
    print(con)


# Linear Discriminant Analysis
for solver in ('svd', 'lsqr', 'eigen'):
    lda = LinearDiscriminantAnalysis(solver=solver)
    lda.fit(X_train, Y_train)
    lda_score = round(lda.score(X_train, Y_train) * 100, 2)
    print('LinearDiscriminantAnalysis Score: \n', lda_score)

    predictions_lda = lda.predict(X_test)

    # Accuracy Score of the LDA
    print("Accuracy Score of LDA is:")
    print(accuracy_score(Y_test, predictions_lda))

    print("Classification Report of LDA is:")
    print(classification_report(Y_test, predictions_lda))
    conf = confusion_matrix(Y_test, predictions_lda)
    print(conf)



# Logistic Regression
lr = LogisticRegression(C=1)
lr.fit(X_train, Y_train)
lr_score = round(lr.score(X_train, Y_train) * 100, 2)
print('Logistic Regression: \n', lr_score)
predictions_lr = lr.predict(X_test)

# Accuracy Score of the LR
print("Accuracy Score of LR is:")
print(accuracy_score(Y_test, predictions_lr))

print("Classification Report OF LR is :")
print(classification_report(Y_test,predictions_lr))
conf1 = confusion_matrix(Y_test,predictions_lr)
print(conf1)

# LR on C=0.01
lr01 = LogisticRegression(C=0.01)
lr01.fit(X_train,Y_train)
lr_score = round(lr01.score(X_train, Y_train) * 100, 2)
print('Logistic Regression on C=0.01: \n', lr_score)
predictions_lr = lr01.predict(X_test)
# Accuracy Score of the LR
print("Accuracy Score of LR on C=0.01 is:")
print(accuracy_score(Y_test, predictions_lr))

# LR on C=0.001
lr001 = LogisticRegression(C=0.001)
lr001.fit(X_train,Y_train)
lr_score = round(lr001.score(X_train, Y_train) * 100, 2)
print('Logistic Regression on C=0.001: \n', lr_score)
predictions_lr = lr001.predict(X_test)
# Accuracy Score of the LR
print("Accuracy Score of LR on C=0.001 is:")
print(accuracy_score(Y_test, predictions_lr))

# LR on C=100
lr100 = LogisticRegression(C=100)
lr100.fit(X_train,Y_train)
lr_score = round(lr100.score(X_train, Y_train) * 100, 2)
print('Logistic Regression on C=100: \n', lr_score)
predictions_lr = lr100.predict(X_test)
# Accuracy Score of the LR
print("Accuracy Score of LR on C=100 is:")
print(accuracy_score(Y_test, predictions_lr))

# Plot to compare the effect of regularization parameter C with different values

monoclonal_features = [x for i,x in enumerate(X.columns) if i!=8]
plt.figure(figsize=(8,6))
plt.plot(lr.coef_.T, 'o', label="C=1")
plt.plot(lr100.coef_.T, '^', label="C=100")
plt.plot(lr01.coef_.T, '^', label="C=0.01")
plt.plot(lr001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(X_train.shape[1]), monoclonal_features, rotation=90)
plt.hlines(0, 0, X_train.shape[1])
plt.ylim(-1, 1)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.savefig('log_coef')
plt.close()

# K Nearest Neighbor
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 51)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, Y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, Y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
#plt.savefig('knn_compare_model')
#plt.show()

count = 0
nob = 50
for i in range(1, nob):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    knn_score = round(knn.score(X_train, Y_train) * 100, 2)
    print('KNN Score for number of neighbours {} is {}: \n'.format(i,knn_score))

    #Predict Output
    predicted = knn.predict(X_test)
    
    # Accuracy Score of KNN
    if i > 10:
        a = accuracy_score(Y_test, predicted)
        if i is 18:
            print("Classification Report :")
            print(classification_report(Y_test, predicted))
            conf2 = confusion_matrix(Y_test, predicted)
            print(conf2)
        count = count+a
        print('Accuracy Score for KNN with number of neighbours {} is {}: \n'.format(i, a))
    else:
        print('Accuracy Score for KNN with number of neighbours {} is {}: \n'.format(i, accuracy_score(Y_test, predicted)))
    
print('The average accuracy is {}'.format((count/(nob-10))))    


# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
nb_score = round(nb.score(X_train, Y_train) * 100, 2)
print('Naive Bayes Score: \n', nb_score)
prediction_nb = nb.predict(X_test)

# Accuracy Score of Naive Bayes
print("Accuracy Score  of NB is:")
print(accuracy_score(Y_test, prediction_nb))

print("Classification Report OF NB is :")
print(classification_report(Y_test,prediction_nb))
conf3 = confusion_matrix(Y_test,prediction_nb)
print(conf3)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, Y_train)
print("Accuracy of Decision Tree classifier on training set: {:.3f}".format(tree.score(X_train, Y_train)))

predictions_dt = tree.predict(X_test)

# Accuracy Score of DT
print("Accuracy Score on prediction is:")
print(accuracy_score(Y_test, predictions_dt))
# Classification Report of DT
print("Classification Report on DTC :")
print(classification_report(Y_test,predictions_dt))
conf4 = confusion_matrix(Y_test,predictions_dt)
print(conf4)

def plot_feature_importances_monoclonal(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), monoclonal_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_monoclonal(tree)
plt.savefig('feature_importance')

# Random Forest
rf = RandomForestClassifier(max_depth=3,n_estimators=100, random_state=0)
rf.fit(X_train, Y_train)
print("Accuracy of Random Forest Classifier on training set: {:.3f}".format(rf.score(X_train, Y_train)))


predictions_rf = rf.predict(X_test)

# Accuracy Score of LG
print("Accuracy Score on prediction is:")
print(accuracy_score(Y_test, predictions_rf))
# Classification Report
print("Classification Report on RFC :")
print(classification_report(Y_test,predictions_rf))
conf5 = confusion_matrix(Y_test,predictions_rf)
print(conf5)

plot_feature_importances_monoclonal(rf)
plt.savefig('feature_importance_rfc')
