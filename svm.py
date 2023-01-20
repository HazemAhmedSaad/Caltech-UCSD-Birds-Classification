# #importing
import pickle
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


lsvc = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=10, tol=0.0001,
          verbose=1)
#validation accuracy
lsvc = lsvc.fit(X_train, y_train)
score = lsvc.score(X_test, y_test)
print("Score: ", score)

matrix = plot_confusion_matrix(lsvc, X_test, y_test,cmap=plt.cm.Blues,normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show()