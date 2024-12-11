==README_START==============================================================
============================================================================
Operating system: MacOS (Sequoia 15.1)
Language: Python
Version: 3.10.4
Libraries / Packages: torch (2.4.1), torchvision.transforms, PIL.image (10.2.0), numpy (1.26.4), glob, os, re (2.2.1), matplotlib.pyplot, matplotlib.pylab (1.26.4), pickle
============================================================================



(X, labelBatch) = genEE364PetImageStack(inputImageCount, imgHeight, imgWidth, colorChannels, use, deviceType)

	*** This function is found in "pet_classifier_modules.ipynb"
	*** Generates a tensor of "inputImageCount" randomly selected cat and dog images

	INPUTS:
	* inputImageCount: Specify the desired number of input images (observations)
	* imgHeight: Specify the height of each input image. imgHeight has value "64" by default
	* imgWidth: Specify the width of each input image. imgWidth has value "64" by default
	* colorChannels: Specify the number of color channels in each input image. colorChannels has value "1" by default (grayscale)
	* use: Specify whether images are sampled from “testdataset/catsfolder" and "testdataset/dogsfolder" (if use = "test") or "traindataset/catsfolder" and "traindataset/dogsfolder" (if use = "train")
	* deviceType: PyTorch device object. Specify device type as “cpu” or “cuda”. deviceType has value torch.device(“cpu”) by default

	OUTPUTS:
	* X: A tensor of cat/dog images with dimensions (inputImageCount x colorChannels x imgHeight x imgWidth)
	* labelBatch: A tensor of true image labels corresponding to the images in X

	VERY IMPORTANT!!!!
	*************************************************************************************************************************
	- PLEASE PLACE TEST IMAGES IN A FOLDER "testdataset/catsfolder" and "testdataset/dogsfolder"
	- If relevant, PLEASE PLACE TRAINING IMAGES IN A FOLDER "traindataset/catsfolder" and "traindataset/dogsfolder"
	*************************************************************************************************************************





pet_classifier_training(datasetSize, epochs, pretrained)

	*** This function is found in "pet_classifier_training.ipynb"
	*** Trains the model and stores the trained weights and biases in ".pth" files found in "params/parametersCNN/" and "params/parametersMLP/"
	*** While training, this function prints the epoch, training loss, and training accuracy	
	*** When done training, this function prints the mean training loss and saves a plot of training loss vs epoch in a file "lossPlot.pdf"

	INPUTS:
	* datasetSize: Specify total number of cat/dog images to train model on. Images are randomly sampled from "traindataset/catsfolder" and "traindataset/dogsfolder". datasetSize has value "800" by default
	* epochs: Specify number of epochs/iterations to train model for. epochs has value "600" by default
	* pretrained: Boolean value specifying whether the model is pretrained (and you would like to continue training it) or not (you would like to begin from scratch). pretrained has default value "False". If pretrained = True, the model loads the pretrained weights and biases from the folders "params/parametersCNN/" and "params/parametersMLP/"

	VERY IMPORTANT!!!!
	*************************************************************************************************************************
	- Although not used, since it is a project requirement, this function provides ".pkl" versions of the trained model weights and biases in the folders "pet_classifier_trainedModel_NOT_USED/parametersCNN" and "pet_classifier_trainedModel_NOT_USED/parametersMLP"
	*************************************************************************************************************************





yguess, pred = pet_classifier(X)

	*** This function is found in "pet_classifier.ipynb" and "pet_classifier.py"
	*** Makes predictions for the labels of X based on the model weights and biases in ".pth" files found in "params/parametersCNN/" and "params/parametersMLP/"
	
	INPUTS:
	* X: A tensor of cat/dog images with dimensions (inputImageCount x colorChannels x imgHeight x imgWidth)

	OUTPUTS:
	* yguess: A tensor of image label predictions corresponding to the images in X
	* pred: A tensor of classification probabilities output by the model (>=0.5 means a "dog" prediction, <0.5 means a "cat" prediction)

	VERY IMPORTANT!!!!
	*************************************************************************************************************************
	- Please note that the model chiefly makes predictions based on the ".pth" model weights and biases found in "params/parametersCNN/" and "params/parametersMLP/"
	- However, since it is a project requirement, I have also provided the ".pkl" versions of these model weights and biases in the folders "pet_classifier_trainedModel/parametersCNN/" and "pet_classifier_trainedModel/parametersMLP/"
	- If you would like to try running the model using weights and biases that you have just trained, please go into "pet_classfier.ipynb" and change the variables cnnParamterPath and mlpParamterPath to cnnParamterPath = "params/parametersCNN/"  and  mlpParamterPath = "params/parametersMLP/", respectively
	*************************************************************************************************************************





calculate_accuracy(dataset_size, X, label_batch, pred, display_results)

	*** This function is found in "pet_classifier_modules.ipynb"
	*** Displays the total number of correct predictions, calculates the accuracy of the model (as a percentage), and displays each image of X with its corresponding prediction and true label
	
	INPUTS:
	* dataset_size: The number of input images (observations)
	* X: A tensor of cat/dog images with dimensions (inputImageCount x colorChannels x imgHeight x imgWidth)
	* label_batch: A tensor of true image labels corresponding to the images in X
	* pred: A tensor of classification probabilities output by the model (>=0.5 means a "dog" prediction, <0.5 means a "cat" prediction)
	* display_results: When "True", the function will display each image of X with its corresponding prediction and true label. When "False", no images are displayed
	


pth_to_pkl(source_folder, target_folder):
	*** This function is found in "pet_classifier_modules.ipynb"
	*** Converts all files in the "source_folder" from ".pth" files to ".pkl" files
	*** Can be used to convert model parameters in "params/parametersCNN/" and "params/parametersMLP/" as per the project requirements


==README_END===============================================================
