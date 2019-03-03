import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# create a pandas dataframe
train = pd.read_csv('/Volumes/DorukHDD/Machine_Learning_CS464/project_data/datasets/latest/train_4800.csv', names=["File Names",'Cardiothorasic Ratio','x', 'y' ,'Label'])
test = pd.read_csv('/Volumes/DorukHDD/Machine_Learning_CS464/project_data/datasets/latest/test_1200.csv', names=['File Names', 'Cardiothorasic Ratio', 'x', 'y', 'Label'])

# separate the dataframe to labels and features
train_features = train.iloc[:, 1:-1]
train_labels = train.iloc[:, -1]
test_features = test.iloc[:, 1:-1]
test_labels = test.iloc[:, -1]

dataset_features = pd.concat([train_features, test_features], ignore_index=True)
dataset_labels = pd.concat([train_labels, test_labels], ignore_index=True)

# normalize all the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_features[["Cardiothorasic Ratio","x","y"]] = scaler.fit_transform(dataset_features[["Cardiothorasic Ratio","x","y"]])

X_train, X_test, y_train, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.2)
X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train.dropna()
y_test = y_test.dropna()

parameters = {  'kernel': ['rbf'],
                'decision_function_shape':['ovo'],
                'gamma':[(lambda x: 2**x)(x) for x in range(-10,15)],
                'C':[ (lambda x: 2**x)(x) for x in range(-7,12)]}

# train svm for cardiomegaly classfication

# test_features = test.iloc[:, 1:-1]
# test_labels = test.iloc[:, -1]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print('Prediction:')
    print(y_pred)
    print('Accuracy Score:')
    print(accuracy_score(y_test, y_pred))
    print()

    # calculate & plot confusion matrix
    cnf_m = confusion_matrix(y_true, y_pred)
    plt.figure()
    class_names = ['No Finding', 'Emphysema', 'Cardiomegaly']
    plot_confusion_matrix(cnf_m, classes=class_names, title='Confusion matrix, without normalization')
    plt.show()


