from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def print_results(algorithm, x_test, y_test, prediction):
	print (algorithm)
	print (confusion_matrix(y_test, prediction))
	print (classification_report(y_test, prediction))

cancer = load_breast_cancer()

x = cancer['data']
y = cancer['target']

x_train, x_test, y_train, y_test = train_test_split(x, y)

# reshaping data 
x_train.reshape(-1, 1)
y_train.reshape(-1, 1)
x_test.reshape(-1, 1)
y_test.reshape(-1, 1)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(x_train, y_train)

mlp_prediction = mlp.predict(x_test)
print_results("MLP", x_test, y_test, mlp_prediction)

dec = tree.DecisionTreeClassifier()
dec = dec.fit(x_train, y_train)
dec_prediction = dec.predict(x_test)
print_results("Decision Tree", x_test, y_test, dec_prediction)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn_prediction = knn.predict(x_test)
print_results("KNN", x_test, y_test, knn_prediction)

svm = svm.SVC(kernel='linear', C = 1.0)
svm.fit(x_train, y_train)
svm_prediction = svm.predict(x_test)
print_results("SVM", x_test, y_test, svm_prediction)

rndf = RandomForestClassifier()
rndf.fit(x_train, y_train)
rndf_prediction = rndf.predict(x_test)
print_results("Random Forest", x_test, y_test, rndf_prediction)
