# Danny McLaughlin
# Applications of Machine Learning to bypassing CAPTCHAs
import os
import cv2
from PIL import Image
import numpy as np
from sklearn import svm, metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from statistics import mean
import string

classes = (string.ascii_lowercase + '1234567890').split()

def get_yes_no_answer(question):
	while True:
		answer = input(question)
		if ((answer == 'y') or (answer == 'Y') or (answer == 'yes') or (answer == 'Yes') or (answer == 'YES')):
			return True
		elif ((answer == 'n') or (answer == 'N') or (answer == 'no') or (answer == 'No') or (answer == 'NO')):
			return False
		else:
			print('\nPlease enter yes or no\n')

def perform_thresholding(full_images):
	for i in range(len(full_images)):
		img = cv2.adaptiveThreshold(full_images[i],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		full_images[i] = img
	return full_images

def perform_smoothing(full_images):
	techniques = {
		"1" : 'Averaging',
		"2" : 'Gaussian',
		"3" : 'Median',
		"4" : 'Bilateral'
	}
	
	invalid_type = True
	while invalid_type:
		print('Smoothing Techniques\n')
		for key, items in techniques.items():
			print(key," : ",items)
		choice = input("\nChoose a type of smoothing:\n")
		if choice.strip().isdigit():
			if int(choice) == 1:
				for i in range(len(full_images)):
					img = cv2.blur(full_images[i],(5,5))
					full_images[i] = img
				invalid_type = False
			elif int(choice) == 2:
				for i in range(len(full_images)):
					img = cv2.GaussianBlur(full_images[i],(5,5),0)
					full_images[i] = img
				invalid_type = False
			elif int(choice) == 3:
				for i in range(len(full_images)):
					img = cv2.medianBlur(full_images[i],5)
					full_images[i] = img
				invalid_type = False
			elif int(choice) == 4:
				for i in range(len(full_images)):
					img = cv2.bilateralFilter(full_images[i],9,75,75)
					full_images[i] = img
				invalid_type = False
			else:
				print("Need to enter a number from the list")
		else:
			print("Need to enter a number from the list")
	
	return full_images

def perform_morphological(full_images):
	techniques = {
		"1" : 'Erosion',
		"2" : 'Dilation',
		"3" : 'Gradient'
	}
	
	# Use a standard kernel for all morphological transformations
	kernel = np.ones((3,3),np.uint8)
	
	invalid_type = True
	while invalid_type:
		print('Morphologies\n')
		for key, items in techniques.items():
			print(key," : ",items)
		choice = input("\nChoose a type of smoothing:\n")
		if choice.strip().isdigit():
			if int(choice) == 1:
				for i in range(len(full_images)):
					img = cv2.erode(full_images[i],kernel,iterations = 1)
					full_images[i] = img
				invalid_type = False
			elif int(choice) == 2:
				for i in range(len(full_images)):
					img = cv2.dilate(full_images[i],kernel,iterations = 1)
					full_images[i] = img
				invalid_type = False
			elif int(choice) == 3:
				for i in range(len(full_images)):
					img = cv2.morphologyEx(full_images[i], cv2.MORPH_GRADIENT, kernel)
					full_images[i] = img
				invalid_type = False
			else:
				print("Need to enter a number from the list")
		else:
			print("Need to enter a number from the list")


	return full_images

def preprocessing(full_images, filenames):
	y, w, h = 8, 22, 40
	images = []
	labels = []
	
	techniques = {
		"1" : {'name' : 'Thresholding', 'func' : perform_thresholding},
		"2" : {'name' : 'Smoothing', 'func' : perform_smoothing},
		"3" : {'name' : 'Morphological', 'func' : perform_morphological},
		"4" : {'name' : 'None', 'func' : None}
	}
	
	adding_filters = True
	while adding_filters:
		print('Preprocessing\n')
		for key, items in techniques.items():
			print(key," : ",techniques[key]['name'])
		choice = input("\nChoose a type of preprocessing:\n")
		try:
			if int(choice) == len(techniques):
				adding_filters = False
			elif choice in techniques:
				func = techniques[choice]['func']
				full_images = func(full_images)
				# show an example of the processing applied
				cv2.imwrite('test_image.png', full_images[0])
				cv2.imshow("Image with Processing", full_images[0])
				cv2.waitKey()
				os.remove("test_image.png")
				adding_filters = False
			else:
				print("Invalid choice - please select again\n")
		except ValueError:
			print("Please enter a number from the list shown\n")
		

		adding_filters = False
	
	
	for i in range(len(full_images)):
		# x is the only coordinate we need to reset for every image
		x = 28
		# Since we assume the images name is the label, the name should be the 
		# length of characters minus '.png'
		captcha = full_images[i]
		# Create an image for every character
		for char in filenames[i][:-4]:
			# m creates an issue because it differs from the standard char size
			if char == 'm':
				break
			labels.append(char);
			# save the result and move to the next position
			char_image = captcha[y:y+h, x:x+w]
			images.append(char_image)
			x += w

	return images, labels

def get_images(encoded_filepath):
	full_images = []
	filenames = []

	# Iterate through directory given and process all png files
	for file in os.listdir(encoded_filepath):
		filename = os.fsdecode(file)
		if filename.endswith(".png"):
			captcha = cv2.imread(os.path.join(filepath, filename), 0)
			full_images.append(captcha)
			filenames.append(filename)
		else:
			continue

	print("\nDone loading images...\n\n")

	# Perform the distortion preprocessing before breaking up the images
	images, labels = preprocessing(full_images, filenames)

	print(f'CAPTCHA Image array size: {len(images)}')
	print(f'Labels array size: {len(labels)}')

	num_instances = len(images)
	flat_images = np.array(images).reshape((num_instances, -1))
	return flat_images, labels	

def split_train_predict(model, final_dataset, final_y):
	X_train, X_test, y_train, y_test = train_test_split(final_dataset, final_y, test_size=0.2)
	model.fit(X_train, y_train)
	pred_y = model.predict(X_test)
	return y_test, pred_y

def do_cross_validation(model, dataset, labels):
	choosing_k = True
	while choosing_k:
		pick = input('Enter the number of folds:\n')
		if pick.strip().isdigit():
			if int(pick) > 10:
				answer = get_yes_no_answer('Are you sure? That\'s a lot of folds...\n')
				if not answer:
					print("Ok let's pick another one")
					continue
			results = cross_val_score(model, dataset, labels, cv=int(pick), scoring='accuracy')
			choosing_k = False
		else:
			print("You need to enter an integer")
			
	return results

def linear_svc(final_dataset, final_y):
	# Linear SVC Model
	invalid = True
	while (invalid):
		c = input("Enter a value for c:\n")
		try:
			c = float(c)
			if c > 0:
				invalid = False
			else:	
				print(" - Please enter a positive value - \n")
		except ValueError:
			print(" - Please enter a number - \n")
	invalid = True
	while (invalid):
		m = input("Enter a value for max_iter:\n")
		try:
			m = int(m)
			if c > 0:
				invalid = False
			else:	
				print(" - Please enter a positive value - \n")
		except ValueError:
			print(" - Please enter a number - \n")

	cross_val = get_yes_no_answer('\nWould you like to use Kfold Cross Validation? y/n\n')
	linear_svc = svm.LinearSVC(C=c, random_state=0, max_iter=m)
	if cross_val:
		results = do_cross_validation(linear_svc, final_dataset, final_y)
		print("Accuracy : ",mean(results))
	else:
		linear_svc_actual, linear_svc_pred = split_train_predict(linear_svc, final_dataset, final_y)
		print("Accuracy : ",metrics.accuracy_score(linear_svc_actual, linear_svc_pred))
		evaluate_whole_captchas(linear_svc_actual, linear_svc_pred)

def rbf_svc(final_dataset, final_y):
	# RBF SVC Model
	invalid = True
	while (invalid):
		c = input("Enter a value for c:\n")
		try:
			c = float(c)
			if c > 0:
				invalid = False
			else:	
				print(" - Please enter a positive value - \n")
		except ValueError:
			print(" - Please enter a number - \n")
		
	cross_val = get_yes_no_answer('\nWould you like to use Kfold Cross Validation? y/n\n')
	svmm = svm.SVC(kernel='rbf', C=c)
	if cross_val:
		results = do_cross_validation(svmm, final_dataset, final_y)
		print("Accuracy : ",mean(results))
	else:
		svm_actual, svm_pred = split_train_predict(svmm, final_dataset, final_y)
		print("Accuracy : ",metrics.accuracy_score(svm_actual, svm_pred))
		evaluate_whole_captchas(svm_actual, svm_pred)

def sgd_classifier(final_dataset, final_y):
	# SGD Classifier Model
	invalid = True
	while (invalid):
		i = input("Enter a value for max iterations:\n")
		try:
			i = int(i)
			if i > 0:
				invalid = False
			else:	
				print(" - Please enter a positive value - \n")
		except ValueError:
			print(" - Please enter a number - \n")

	cross_val = get_yes_no_answer('\nWould you like to use Kfold Cross Validation? y/n\n')
	sgdc = SGDClassifier(max_iter=i)
	if cross_val:
		results = do_cross_validation(sgdc, final_dataset, final_y)
		print("Accuracy : ",mean(results))
	else:
		sgd_actual, sgd_pred = split_train_predict(sgdc, final_dataset, final_y)
		print("Accuracy : ",metrics.accuracy_score(sgd_actual, sgd_pred))
		evaluate_whole_captchas(sgd_actual, sgd_pred)

def knn_classifier(final_dataset, final_y):
	# KNN Classifier
	invalid = True
	while (invalid):
		n = input("Enter a value for n:\n")
		try:
			n = int(n)
			if n > 0:
				invalid = False
			else:	
				print(" - Please enter a positive value - \n")
		except ValueError:
			print(" - Please enter a number - \n")

	knn = KNeighborsClassifier(n_neighbors=n)
	cross_val = get_yes_no_answer('\nWould you like to use Kfold Cross Validation? y/n\n')
	if cross_val:
		results = do_cross_validation(knn, final_dataset, final_y)
		print("Accuracy : ",mean(results))
	else:
		knn_actual,knn_pred = split_train_predict(knn, final_dataset, final_y)
		print("Accuracy : ",metrics.accuracy_score(knn_actual, knn_pred))
		evaluate_whole_captchas(knn_actual, knn_pred)

def nearest_centroid(final_dataset, final_y):
	# NearestCentroid Classifier
	centroid = NearestCentroid()
	cross_val = get_yes_no_answer('\nWould you like to use Kfold Cross Validation? y/n\n')
	if cross_val:
		results = do_cross_validation(centroid, final_dataset, final_y)
		print("Accuracy : ",mean(results))
	else:
		centroid_actual, centroid_pred = split_train_predict(centroid, final_dataset, final_y)
		print("Accuracy : ",metrics.accuracy_score(centroid_actual, centroid_pred))
		evaluate_whole_captchas(centroid_actual, centroid_pred)

def decision_tree(final_dataset, final_y):
	# Decision Tree Classifier
	invalid = True
	while invalid:
		depth = input("Enter a maximum tree depth:\n")
		if depth.strip().isdigit():
			if int(depth) > 0:
				invalid = False
			else:
				print("Number needs to be greater than 0\n")
		else:
			print("Need to enter an integer greater than 0")

	tree = DecisionTreeClassifier(max_depth=int(depth))
	cross_val = get_yes_no_answer('\nWould you like to use Kfold Cross Validation? y/n\n')
	if cross_val:
		results = do_cross_validation(tree, final_dataset, final_y)
		print("Accuracy : ",mean(results))
	else:
		tree_actual, tree_pred = split_train_predict(tree, final_dataset, final_y)
		print("Accuracy : ",metrics.accuracy_score(tree_actual, tree_pred))
		evaluate_whole_captchas(tree_actual, tree_pred)

def mlp_classifier(final_dataset, final_y):
	# Multi Layer Perceptron
	solvers = {
		"1": {'name' : "Newton Methods", 'type' : 'lbfgs'},
		"2": {'name' : "Stochastic Gradient Descent", 'type' : 'sgd'},
		"3": {'name' : "Adam", 'type' : 'adam'}
	}
	
	# Loop until  a valid solver is chosen
	invalid_solver = True
	while( invalid_solver ):
		print('\nSolvers :')
		for key, value in solvers.items():
			print(key," : ",solvers[key]['name'])
		choice = input("\nChoose a solver to use:\n")
		try:
			if choice in solvers:
				solver = solvers[choice]['type']
				invalid_solver = False
			else:
				print("Invalid choice - please select again\n")
		except ValueError:
			print("Please enter a number from the list shown\n")
	
	clf = MLPClassifier()#solver='lbfgs', hidden_layer_sizes=(5, 2), random_state=1)
	cross_val = get_yes_no_answer('\nWould you like to use Kfold Cross Validation? y/n\n')
	if cross_val:
		results = do_cross_validation(clf, final_dataset, final_y)
		print("Accuracy : ",mean(results))
	else:
		mlp_actual, mlp_pred = split_train_predict(clf, final_dataset, final_y)
		print("Accuracy : ",metrics.accuracy_score(mlp_actual, mlp_pred))
		evaluate_whole_captchas(mlp_actual, mlp_pred)

def choose_algorithm(final_dataset, final_y):
	algorithms = {
		"1": {"Linear SVC" : linear_svc},
		"2": {"RBF SVC" : rbf_svc},
		"3": {"SGD Classifier" : sgd_classifier},
		"4": {"KNearest Neighbors" : knn_classifier},
		"5": {"Nearest Centroid" : nearest_centroid},
		"6": {"Decision Tree" : decision_tree},
		"7": {"Multi Layer Perceptron" : mlp_classifier},
		"8": {"Quit": "null"}
	}

	invalid_algorithm = True
	# Loop until a valid algorithm is chosen
	while( invalid_algorithm ):
		print('\nAlgorithms :')
		for key, value in algorithms.items():
			for name,  func in value.items():
				print(key," : ",name)
		choice = input("\nChoose an algorithm to run on the dataset:\n")
		try:
			if int(choice) == len(algorithms):
				invalid_algorithm = False
			elif choice in algorithms:
				d = algorithms.get(choice, {})
				func = next(iter(d.values()))
				func(final_dataset, final_y)
				invalid_algorithm = False
			else:
				print("Invalid choice - please select again\n")
		except ValueError:
			print("Please enter a number from the list shown\n")

def evaluate_whole_captchas(y_test, predicted):
	# Evaluate performance of multiple characters i.e. a single captcha
	whole_images = np.stack((y_test, predicted), axis=-1)
	iter = len(y_test)/5
	correct_count = 0

	for i in range(int(iter)):
		correct = True
		for j in range(5):
			index = 5*i + j
			if(len(np.unique(whole_images[index])) != 1):
				correct = False
			
		if(correct):
			correct_count += 1

	print("\nCAPTCHA accuracy: ",correct_count/iter)

# Get file path from user
# Assumes file names are CAPTCHA values
filepath = input("Enter filepath for data location:\n")
encoded_filepath = os.fsencode(filepath);
# Create a list of images and list of labels
# Mapped by index
images, labels = get_images(encoded_filepath)

# Execution proceeds internally
choose_algorithm(images, labels)





