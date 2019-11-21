import numpy as np
import sys, re, os

# ML library imports
try:
    from sklearn import svm
except:
    print("Please install the module 'sklearn' for SVM!")
    print("pip install sklearn")
    sys.exit(-1)

from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# load the speech responses
def loadResponse(pathname):
    path = os.path.join("./", pathname)
    if not os.path.isdir(path):
        print('Given path {} not found'.format(path))
        sys.exit(-1)
    # max speech sequence
    max_seq = 3000
    samples = []
    labels = []
    #	record_num = np.zeros(10)
    #	limit=2000
    for fn in os.listdir(path):
        if fn[0] == '.':
            continue
        if fn[0] == 'S':
            continue
        if fn[0] == 'f':
            continue
        if fn[0] == 'c':
            continue
        filename = os.path.join(pathname, fn)
        m = re.findall('[0-9]+', fn)
        if os.path.isfile(filename):
            label = float(m[1])
            #			index=int(label)
            #			if record_num[index]>limit:
            #				continue
            #			record_num[index]+=1
            labels.append(label)
            with open(filename, 'r') as f:
                data = np.loadtxt(f, dtype=int)
                indices = np.argwhere(data == -1)
                # flatten the list:
                indices = [item for sublist in indices for item in sublist]
                nrows = len(indices) - 1

                # reservoir response matrix:
                mat = np.zeros([nrows, max_seq])
                for i in range(nrows):
                    fired_times = data[indices[i] + 1: indices[i + 1]]
                    if len(fired_times) == 1 and fired_times[0] == -99:
                        continue
                    for t in fired_times:
                        assert t > 0
                        if t >= max_seq:
                            continue
                        mat[i, t] = 1
                #	for row in mat:
                #		print(len(row))
                mat = sum(mat.transpose())
                mat_new = np.array(mat)
                #			print(mat_new.shape)
                mat = mat.flatten()
                samples.append(mat)
    return samples, labels


def cvSVM(s, y, nfold):
    samples_scaled = preprocessing.scale(s)
    x = np.array(samples_scaled)

    # sel = VarianceThreshold()
    # X2=sel.fit_transform(X1)
    # print(X2.shape)
    # X_new = SelectKBest(chi2, k=400).fit_transform(X2, y)
    # print(X_new.shape)
    params = [{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}, ]
    clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), params, cv=nfold, iid=False)
    clf.fit(x, y)
    scores = cross_val_score(clf, x, y, cv=nfold)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()


def traintestSVM(train_samples, train_labels, test_samples, test_labels):

    clf = svm.SVC(kernel='rbf', C=0.0001, gamma='auto', decision_function_shape='ovr')

    train_scaled = preprocessing.scale(train_samples)
    test_scaled = preprocessing.scale(test_samples)
    clf.fit(train_scaled, train_labels)
    predict = clf.predict(test_scaled)
    accuracy = 0
    for i in range(len(test_labels)):
        accuracy += test_labels[i] == predict[i]
    accuracy = accuracy / float(len(test_labels))

    return accuracy
