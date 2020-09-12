# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
import sklearn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import VotingClassifier

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    matrix=ConfusionMatrix(y_true,y_pred)
    right_pred=0
    size=np.unique(y_true).size
    for i in range(size):
        for j in range(size):
            if i==j:
                right_pred+=matrix[i][j]
        
    return right_pred/y_pred.shape[0]

def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    matrix=ConfusionMatrix(y_true,y_pred)
    res=[]
    size=np.unique(y_true).size
    for i in range(size):
        FN=0
        for j in range(size):
            if i==j:
                TP=matrix[i][j]
            else:
                FN=FN+matrix[i][j]
        res.append(TP/(TP+FN))
    return np.average(res)

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    matrix=ConfusionMatrix(y_true,y_pred).T
    res=[]
    size=np.unique(y_true).size
    for i in range(size):
        FP=0
        for j in range(size):
            if i==j:
                TP=matrix[i][j]
            else:
                FP=FP+matrix[i][j]
        res.append(TP/(TP+FP))

    return np.average(res)

def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    final_wcss = 0
    for cluster in Clusters:
        dist = np.linalg.norm(np.mean(cluster, axis=0) - cluster, axis=1)
        final_wcss += np.sum(dist**2) # sum of squared distances
    return final_wcss

def ConfusionMatrix(Y_true,Y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    size = np.unique(Y_true).size
    con = (Y_true*size) + Y_pred
    hist = np.histogram(con, bins=np.arange(size+1,(size*size)+(size+1)+1))
    return hist[0].reshape(size,size)

def KNN(X_train,X_test,Y_train, K):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: numpy.ndarray
    """
    Y_predict = np.zeros(X_test.shape[0])
    
    # loop over X_test for all points
    for i in range(X_test.shape[0]):
        # get current test datapoint
        X_test_cur = X_test[i,]
        
        # broadcast or repeat point to match X_train shape not required 
        # as numpy does this for '-' operation automatically if possible
#       X_test_repeated = np.repeat(np.expand_dims(X_test_cur, axis = 0), X_train.shape[0], axis = 0)
        
        # distances from all points, found np.linalg.norm function to take 10-15 extra seconds for 80% of X at runtime compared to this
        X_test_diff = np.sqrt(np.sum((X_test_cur - X_train) ** 2, axis=1))
        
        # stack the distance with corresponding class for the data point and transpose it to make it mx2 size
        neighbors = np.stack((X_test_diff, Y_train)).transpose()
        
        # apply sorting based on the distance column i.e. column 0
        neighbors = neighbors[neighbors[:,0].argsort()]   #https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/2828121#2828121
        
        # get the k closest points
        neighbors = neighbors[:K]
        
        # transpose to get all classes in 1 axis, convert its type to int, get binary count of classes, and find the
        # class with most count
        Y_predict[i] = np.argmax(np.bincount(neighbors.transpose()[1].astype(int)))
        
    return Y_predict

def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    # total no. of decision trees used in bagging for random forest
    number_of_trees = 10
    
    # no. of samples to consider for bootstrapping
    number_of_samples = int(X_train.shape[0]/6)
    
    # minimum information gain value below which we stop splitting further
    allowed_info_gain = 0.001
    
    # list of decision trees
    decision_trees = []

    # initiallize and train all the decision trees
    for i in range(number_of_trees):
        
        # bootstrapping the samples
        rows_indices = np.random.choice(range(X_train.shape[0]), size=number_of_samples, replace=True)
        
        # initiallize a decision tree and train it on the bootstrapped rows
        decision_tree = Decision_Tree(allowed_info_gain=allowed_info_gain)
        decision_tree.fit(X_train[rows_indices,],Y_train[rows_indices])
        
        # append the trained tree to the list
        decision_trees.append(decision_tree)
    
    # initiallize prediction array with trees in row and predictions as columns. shape = (no. of trees, no. or rows)
    Y_pred_all = np.zeros((number_of_trees,X_test.shape[0]))
    
    # bagging
    # have predictions from all the decision trees
    for treeI in range(number_of_trees):
        Y_pred_all[treeI] = decision_trees[treeI].predict(X_test)
    
    # final prediction array
    Y_pred = np.zeros(X_test.shape[0])
    
    # predict the most common label among all the trees as the final label
    for i in range(Y_pred_all.shape[1]):
        Y_pred[i] = np.argmax(np.bincount(Y_pred_all[:,i].astype(int)))
    
    return Y_pred
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    # center columns by subtracting column means
    centered_cols = X_train - np.mean(X_train,axis=0)
    
    # calculate covariance matrix of centered matrix
    covariance_mat = np.cov(centered_cols.T)
    
    # eigendecomposition of covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(covariance_mat)

    # make eigen pairs of eigen values and vectors
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
    
    # Sort the (eigen_values, eigen_vectors) tuples from high to low eigen values
    eigen_pairs.sort(key=lambda x: x[0],reverse=True)
    
    # get the top N vectors from the tuples
    eigen_vector = np.array([(eigen_pairs[i][1]) for i in range(N)])
    return eigen_vector.dot(centered_cols.T).T

def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    centroids = np.zeros((N, X_train.shape[1]))
    indices = np.random.choice(X_train.shape[0], size=N, replace="False")
    for i in range(len(indices)):
        centroids[i] = X_train[indices[i]]

    X_test_diff = np.zeros((X_train.shape[0], N))

    classes = []

    while True:
        countCentroid = 0

        for centroid in centroids:            
            
            # distances from all points to centroid, centroid is broadcasted to X_train's shape automatically by numpy before subtracting
            # np.linalg.norm was giving bigger difference between sklearn's inertia and wcss value of this, 
            # so distance is found this way instead of np.linalg.norm
            X_test_diff[:, countCentroid] = np.sqrt(np.sum((centroid - X_train) ** 2, axis=1))

            countCentroid += 1

        # take indices of minimum distance among all distances to the centroids
        classes = np.argmin(X_test_diff, axis=1)

        # make copy of centroids before updating them, so as to check if they're changing in particular iteration or not
        old_centroids = deepcopy(centroids)

        # calculate new centroids
        for o in range(N):
            centroids[o] = np.mean(X_train[classes == o], axis=0)

        # check if new centroids are same as old_centroids, if yes, we're done
        if np.linalg.norm(centroids - old_centroids) == 0:
            break

    # make dummy list of numpy.ndarray
    clusters = [np.zeros((1, X_train.shape[1]))]*11

    # populate return list with X_train for respective clusters
    for o in range(N):
        clusters[o] = X_train[classes == o]

    return clusters

def SklearnSupervisedLearning(X_train,Y_train,X_test, Y_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    # order of algorithms: SVM, Logistic Regression, Decision Tree, KNN
    Y_preds = []
    
    # best parameters found in grid search ran on collab are used for below models to get best results

    # SVM
    model_svm = sklearn.svm.SVC(C=200,kernel='linear')
    model_svm = model_svm.fit(X_train,Y_train)
    Y_preds.append(model_svm.predict(X_test))
    print("Accuracy Score of SVM Model:",Accuracy(Y_test,Y_preds[0]))

    # Logistic Regression
    model_lr = sklearn.linear_model.LogisticRegression(multi_class='auto')
    model_lr.fit(X_train, Y_train)
    Y_preds.append(model_lr.predict(X_test))
    print("Accuracy Score of Logistic Regression Model:",Accuracy(Y_test,Y_preds[1]))

    # Decision Tree
    model_tree = sklearn.tree.DecisionTreeClassifier(max_depth=20)
    model_tree.fit(X_train, Y_train)
    Y_preds.append(model_tree.predict(X_test))
    print("Accuracy Score of Decision Tree Model:",Accuracy(Y_test,Y_preds[2]))
    # KNN
    model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, algorithm="auto", p=1)
    model_knn.fit(X_train, Y_train)
    Y_preds.append(model_knn.predict(X_test))
    print("Accuracy Score of KNN Model:",Accuracy(Y_test,Y_preds[3]))
    return Y_preds

def SklearnVotingClassifier(X_train,Y_train,X_test,Y_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: just an array (prediction) 
    """
    model_svm = sklearn.svm.SVC(C=200,kernel='linear')
    model_lr = sklearn.linear_model.LogisticRegression()
    model_tree = sklearn.tree.DecisionTreeClassifier(max_depth=20)
    model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, algorithm="auto", p=1)

    model_voting_classifier = sklearn.ensemble.VotingClassifier(estimators=[('svm', model_svm), ('lr', model_lr), ('tree', model_tree), ('knn', model_knn)], voting='hard')   
    model_voting_classifier = model_voting_classifier.fit(X_train, Y_train)  
    Y_pred = model_voting_classifier.predict(X_test)
    print("Accuracy Score of Voting classifier model:",Accuracy(Y_test,Y_pred))
    return Y_pred

def visualize_confusion_matrices(confusion_matrices):
    # confusion_matrices are coming in order SVM, Logistic Regression, Decision Tree, KNN, voting classifier
    #https://matplotlib.org/3.1.3/tutorials/colors/colormaps.html

    f, plts = plt.subplots(3,2,figsize=(18,20))
    f.suptitle('CONFUSION MATRICES')
    fill_conf_mat_plot(plts[0,0], confusion_matrices[0], "SVM")
    fill_conf_mat_plot(plts[0,1], confusion_matrices[1], "Logistic Regression")
    fill_conf_mat_plot(plts[1,0], confusion_matrices[2], "Decision Tree")
    fill_conf_mat_plot(plts[1,1], confusion_matrices[3], "KNN")
    fill_conf_mat_plot(plts[2,0], confusion_matrices[4], "Voting Classifier")
    plts[2,1].axis('off')

def fill_conf_mat_plot(cur_plot, confusion_matrix, title):
    cur_plot.matshow(confusion_matrix, cmap='plasma')
    cur_plot.set_title(title)
    for (i, j), val in np.ndenumerate(confusion_matrix):
        cur_plot.text(j, i, val, ha='center', va='center')


# GRID SEARCH
def perform_grid_search(X,Y):

    # SVM Grid Search
    
    model_svm = sklearn.svm.SVC()
    parameters = [{'C': [10, 20, 40, 80, 100, 130, 150, 200], 'kernel': ['linear']}] #,'gamma':[0.1, 0.5, 0.9]
    grid_search_svm = sklearn.model_selection.GridSearchCV(estimator = model_svm, param_grid = parameters, scoring = 'accuracy', return_train_score=True, n_jobs = -1)
    grid_search_svm = grid_search_svm.fit(X, Y)
    
    print("SVM model's best parameters:", grid_search_svm.best_estimator_)
    svm_res = grid_search_svm.cv_results_

    #print(res.keys())
    svm_params = svm_res.get('param_C')
    svm_scores = svm_res.get('mean_test_score')

    # Decision Tree Grid Search
    
    model_tree = sklearn.tree.DecisionTreeClassifier()
#   parameters = [{'splitter': ['best','random'], 'max_depth': [3, 5, 8, 10, 12, 14, 16, 18, 20, 30], 
#                  'min_samples_split':[2,4,10,100] , 'max_features' : [int(np.sqrt(X_train.shape[1]))]}]
    parameters = [{'splitter': ['best'], 'max_depth': [2, 3, 5, 8, 10, 12, 14, 16, 18, 20, 30, 35, 40, 50], 
                   'min_samples_split':[4] , 'max_features' : [int(np.sqrt(X.shape[1]))]}]
    grid_search_dt = sklearn.model_selection.GridSearchCV(estimator = model_tree, param_grid = parameters, scoring = 'accuracy', n_jobs = -1)
    grid_search_dt = grid_search_dt.fit(X, Y)

    print("Decision Tree model's best parameters:", grid_search_dt.best_estimator_)
    dt_res = grid_search_dt.cv_results_

    dt_params = dt_res.get('param_max_depth')
    dt_scores = dt_res.get('mean_test_score')
    
    # KNN Grid Search
    
    model_knn = sklearn.neighbors.KNeighborsClassifier()
#   parameters = [{'n_neighbors': [4, 5, 6, 7, 8, 9, 10, 11], 'weights':['uniform','distance'], 
#                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],'p':[1,2]}]
    parameters = [{'n_neighbors': [ 1, 2, 5, 7, 10, 15, 20, 30, 40], 'weights':['distance'],'p':[1]}]
    grid_search_knn = sklearn.model_selection.GridSearchCV(estimator = model_knn, param_grid = parameters, scoring = 'accuracy', n_jobs = -1)
    grid_search_knn = grid_search_knn.fit(X, Y)

    print("KNN model's best parameters:", grid_search_knn.best_estimator_)
    knn_res = grid_search_knn.cv_results_

    knn_params = knn_res.get('param_n_neighbors')
    knn_scores = knn_res.get('mean_test_score')

    plot_grid_search_graphs(svm_params,svm_scores, dt_params,dt_scores, knn_params, knn_scores)

 
def plot_grid_search_graphs(svm_params, svm_scores, dt_params, dt_scores, knn_params, knn_scores):
    f, plts = plt.subplots(2,2,figsize=(18,16))
    f.suptitle('Accuracy vs parameters')
    plt1 = plts[0,0]
    fill_grid_search_plot(plts[0,0], svm_params, svm_scores, "SVM GridSearch", "C Values")
    fill_grid_search_plot(plts[0,1], dt_params, dt_scores, "Decision Tree GridSearch", "Depth Values")
    fill_grid_search_plot(plts[1,0], knn_params, knn_scores, "KNN GridSearch", "n_neighbors Values")
    plts[1,1].axis('off')

def fill_grid_search_plot(cur_plt, params,scores, title, param_label):
    cur_plt.plot(params,scores)
    cur_plt.scatter(params, scores)
    cur_plt.set_title(title)
    cur_plt.set_xlabel(param_label)
    cur_plt.set_ylabel("Accuracy")
    cur_plt.grid(True)

"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""

# helper methods and classes below

# Decision_Tree is used in random_forest
class Decision_Tree:
    
    def __init__(self, allowed_info_gain=0):
        self.allowed_info_gain = allowed_info_gain
    
    # builds the decision tree sets the root node of the built tree as member of class
    def fit(self, X_train, Y_train):
        # limit of number of features to be considered for split
        self.features_limit = int(np.sqrt(X_train.shape[1]))
        self.root = self.make_decision_tree(X_train,Y_train,1)
    
    # uses the decision tree built to predict the classes of given data points
    def predict(self, X_test):  

        Y_pred = np.zeros(X_test.shape[0])
        for row in range(Y_pred.size):
            Y_pred[row] = self.predict_class(self.root, X_test[row,])
        return Y_pred
        
    # predicts the class of the given single data point
    def predict_class(self, node, X_test):
        if(node.threshold == None):
            return node.possible_class
        if(node.left is not None and X_test[node.feature] < node.threshold):
            return self.predict_class(node.left, X_test)
        if(node.right is not None and X_test[node.feature] > node.threshold):
            return self.predict_class(node.right, X_test)
        return node.possible_class
    
    # recursively builds and returns a binary tree using the Node class
    def make_decision_tree(self, X, Y, depth):

        # find the best guess for a class considering the node as leaf node
        class_counts = np.array([np.sum(Y==float(i)) for i in range(int(np.max(Y))+1)])
        possible_class = np.argmax(class_counts)
        
        root = Node(possible_class)
        
        # enter only if more than 1 classes and more than than 1 row are present
        if(len(class_counts[class_counts>0]) > 1 and Y.size > 1):
            
            # get the best feature and its threshold value
            root.feature, root.threshold = self.get_best_split(X,Y)
            
            # uncomment to see node info that is being made
            #print("node made at depth",depth, "with --", Y.size, "-- rows on feature",root.feature, "with threshold",root.threshold)
            
            # call recursively on the split data if some threshold was returned
            if(root.threshold is not None):    
                X_left, X_right, Y_left, Y_right = self.partition_data(X,Y,root.feature, root.threshold)
                root.left = self.make_decision_tree(X_left, Y_left, depth + 1)
                root.right = self.make_decision_tree(X_right, Y_right, depth + 1)
        return root
    
    # finds and returns the feature and its threshold value for which information gain is highest
    def get_best_split(self, X, Y):
        
        max_info_gain = 0
        threshold = 0
        
        # stops splitting further if only 1 row is present
        if(Y.size == 1):
            return None, None
        feature, threshold = None, None
        
        # pick a list of random features to find the next split point on
        features_indices = np.random.choice(range(X.shape[1]), size=self.features_limit , replace=False)
        
        for i in features_indices:
            
            # get threshold and info gain for current column
            cur_threshold, cur_info_gain = self.get_best_threshold(X, Y, i)
            
            # update the max_info_gain, feature, and threshold value if info gain found was better
            if(max_info_gain < cur_info_gain):
                max_info_gain = cur_info_gain
                feature = i
                threshold = cur_threshold
        
        # if final info gain is below the allowed_info_gain, then no further splits are done
        if(max_info_gain < self.allowed_info_gain):
            return None, None
        return feature, threshold
    
    # find the best threshold for a feature having continuous data
    # returns the best threshold and the max information gain found
    def get_best_threshold(self, X, Y, feature):
        
        # get data only for the given column
        col_data = X[:,feature]
        
        # sort the rows so partition can be done directly when threshold is found
        sorted_indices = col_data.argsort()
        col_data, labels = col_data[sorted_indices], Y[sorted_indices]
        
        # parent gini index value
        p_gini = self.gini_impurity(Y)
        
        max_info_gain = 0
        best_threshold = None
        
        # go over values for the feature for all data points
        for i in range(col_data.shape[0]-1):
            
            # skip if 2 points are same
            if(col_data[i] == col_data[i+1]):
                continue
            
            # average of 2 points
            cur_threshold = (col_data[i] + col_data[i+1])/2
            
            # CHECK Ith value repetition
            # partition the data and find gini value for left and right parts
            Y_left, Y_right = labels[0:i+1], labels[i:Y.size] 
            gini_left = self.gini_impurity(Y_left)
            gini_right = self.gini_impurity(Y_right)

            # find information gain based on current split
            info_gain = p_gini - ((Y_left.size/Y.size) * gini_left + (Y_right.size/Y.size) * gini_right)
            
            # update threshold and information gain if found info gain is higher
            if(max_info_gain < info_gain):
                max_info_gain = info_gain
                best_threshold = cur_threshold

        return best_threshold, max_info_gain
    
    # calculates and returns gini index for the given array of labels
    def gini_impurity(self, Y):
        psum = 0 
        for label in range(int(np.min(Y)), int(np.max(Y)+1)):
            psum += (np.sum(Y==label) / Y.shape[0]) ** 2
        return 1 - psum
    
    # returns partitioned data based on feature and threshold value for that feature
    def partition_data(self, X, Y, feature, threshold):
        left_index_list = X[:,feature] < threshold
        right_index_list = X[:,feature] >= threshold
        X_left, X_right = X[left_index_list], X[right_index_list]
        Y_left, Y_right = Y[left_index_list], Y[right_index_list]
        return X_left, X_right, Y_left, Y_right
    
# Node class is used to create the decision binary tree that is used in random forest
class Node:
    
    def __init__(self, possible_class):
        self.possible_class = possible_class
        self.threshold = 0
        self.feature = 0
        self.left = None
        self.right = None
