import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

breast_cancer = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"])

breast_cancer.head()
breast_cancer.info()

breast_cancer['diagnosis'].unique()

X = breast_cancer[['radius_se', 'concave points_worst']].values 
Y = breast_cancer[['diagnosis']].values

X_test, X_train, Y_test, Y_train = train_test_split(X, Y, test_size=0.3, random_state=0)

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

le = LabelEncoder()

Y_train_le = le.fit_transform(Y_train.ravel())
Y_test_le = le.transform(Y_test.ravel())

lr = LogisticRegression()
lr.fit(X_train, Y_train_le)
Y_pred = lr.predict(X_test)
Y_pred_prob = lr.predict_proba(X_test)

accs = accuracy_score(Y_test_le, Y_pred)
lloss = log_loss(Y_test_le, Y_pred_prob)

print('Single - Accuracy: ' + str(accs) + ' Loss:' + str(lloss))

# analizziamo il decision boundary

def show_bounds(model,X,Y,labels=["Classe 0","Classe 1"], figsize=(12,10)):

    plt.figure(figsize=figsize)

    h = .02

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    X_m = X[Y==1]
    X_b = X[Y==0]
    plt.scatter(X_b[:, 0], X_b[:, 1], c="green",  edgecolor='white', label=labels[0])
    plt.scatter(X_m[:, 0], X_m[:, 1], c="red",  edgecolor='white', label=labels[1])
    plt.legend()


def show_bounds(model, X, Y, labels=["Benigno", "Maligno"], figsize=(8, 6)):
    plt.figure(figsize=figsize)

    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Predict over the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contours
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

    # Plot the training points
    for i, label in enumerate(np.unique(Y)):
        plt.scatter(X[Y == i, 0], X[Y == i, 1], label=labels[i])

    plt.legend()
    plt.show()

show_bounds(lr, X_train, Y_train_le)
show_bounds(lr, X_test, Y_test_le)

# proviamo con tutti i dati del dataset
X = breast_cancer.drop(['diagnosis', 'id'], axis=1).values 
Y = breast_cancer[['diagnosis']].values

X_test, X_train, Y_test, Y_train = train_test_split(X, Y, test_size=0.3, random_state=0)

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

le = LabelEncoder()

Y_train_le = le.fit_transform(Y_train.ravel())
Y_test_le = le.transform(Y_test.ravel())

lr = LogisticRegression()
lr.fit(X_train, Y_train_le)
Y_pred = lr.predict(X_test)
Y_pred_prob = lr.predict_proba(X_test)

accs = accuracy_score(Y_test_le, Y_pred)
lloss = log_loss(Y_test_le, Y_pred_prob)

print('All - Accuracy: ' + str(accs) + ' Loss:' + str(lloss))

show_bounds(lr, X_train, Y_train_le)
show_bounds(lr, X_test, Y_test_le)