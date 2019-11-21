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


def cvSVM(X, y, response_path, nfold):
    filename = response_path + "SVM_ovr_result.txt"
    X = np.array(X)
    #	print(X.shape)
    clf = svm.LinearSVC(C=0.0001, multi_class='ovr')
    #	sel = VarianceThreshold()
    #	X2=sel.fit_transform(X1)
    #	print(X2.shape)
    #	X_new = SelectKBest(chi2, k=400).fit_transform(X2, y)
    #	print(X_new.shape)
    #	params={'C':[1e-5,1e-4,0.001,0.01,0.1,1,10]}
    #	clf=GridSearchCV(svm.SVC(kernel='linear',decision_function_shape='ovr'),params,cv=5)
    #	clf.fit(X,y)
    #	print(clf.best_params_)
    scores = cross_val_score(clf, X, y, cv=5)
    #	print(scores)
    #	print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    #	np.savetxt(filename, scores)
    file_object = open(filename, 'a')
    #	file_object.write(X.shape)
    file_object.write("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    file_object.close()


def trainTestSVM(samples, labels):
    clf = svm.LinearSVC()
    half = int(len(samples) / 2)
    train = samples[: half]
    train_labels = labels[: half]
    test = samples[half:]
    test_labels = labels[half:]

    clf.fit(train, train_labels)
    predict = clf.predict(test)
    accuracy = 0
    for i in range(len(test_labels)):
        accuracy += test_labels[i] == predict[i]

    len_ = len(test_labels)
    print("Accuracy: %0.2f" % (accuracy / float(len_)))


def main():
    response_path = sys.argv[1]
    samples, labels = loadResponse(response_path)
    print(labels)
    print(len(labels))
    # 5 fold cross validation
    cvSVM(samples, labels, response_path, nfold=5)
