#importing and loading the necessary python libraries and the breast cancer dataset provided by Scikit-learnfrom sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

#create new variables for each important set of information that we find useful and assign the attributes in the dataset to those variables
data = load_breast_cancer()
label_names = data["target_names"]
labels = data["target"]
feature_names = data["feature_names"]
features = data["data"]

#We now have values for each set of useful information in the dataset. To better understand our dataset, let’s take a look at our data by printing our class labels, the label for the first data instance, our entity names, and the entity values for the first data instance:
print(label_names)
print("Class label :", labels[0])
print(feature_names)
print(features[0], "\n")

#splitting dataset into 80% training and 20% testing:
train, test, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

#using naive bayes for breast cancer detection
gnb = GaussianNB()
gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds, "\n")
#use the accuracy_score () function provided by Scikit-Learn to determine the accuracy rate of our machine learning classifier
print(accuracy_score(test_labels, preds))
