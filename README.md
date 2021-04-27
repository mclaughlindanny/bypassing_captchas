# bypassing_captchas
  
This application can be run as a single file. The execution process is described below.  
  
Dataset : https://www.kaggle.com/fournierp/captcha-version-2-images  
  
Running the Application  
1) The first step in the process is to enter a path to a directory where all the images are stored as .png images  
2) The next step in the process is choosing a type of distortion preprocessing. Descriptions for each type can be found in the following links.  
  a) Thresholding  - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html  
  b) Smoothing     - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html  
  c) Morphological - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#morphological-ops  
    
3) Following preprocessing an algorithm can be selected. Links to descriptions of each algorithm and their hyperparameters can be found below  
  a) Linear Support Vector Classifier (SVC) - https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html  
  b) Radial Basis Function SVC - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html  
  c) Stochastic Gradient Descent Classifier - https://scikit-learn.org/stable/modules/sgd.html  
  d) K-Nearest Neighbors Classifier - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
  e) Nearest Centroid Classifier - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html  
  f) Decision Tree Classifier - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html  
  g) Multi Layer Perceptron Classifier - https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html  
    
4) After choosing an algorithm you will be asked if you want to perform K-Fold cross validation.  
  a) If you select yes, you will be able to enter a number of folds and a model using your chosen algorithm will be trained and evaluated for accuracy.  
  b) If you select no, your model will be trained once on the given algorithm and an accuracy for individual characters as well as overall CAPTHCAs will be returned.  
  
