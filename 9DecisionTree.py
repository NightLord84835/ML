import pandas as pd
data = pd.read_csv("Admission.csv")

#data['Admitted'] = data['Admitted'].apply(lambda y: 1 if y >= 0.75 else 0)

from sklearn.preprocessing import Binarizer
data['Admitted'] = Binarizer(threshold=0.75).transform(data[['Admitted']])
print(data['Admitted'].astype)
     
x = data[['GRE Score','CGPA']]
y = data['Admitted']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Classification report: ")
print(classification_rep)