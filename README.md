# TrOCR

The IAM folder holds the images from the IAM dataset. The smaller subset of images was taken directly from the unilm IAM test set that is available for download.
The larger set of images followed the GitHub repository link in the paper, obtained the IAM dataset from the source, and went through the processing steps listed
in that GitHub repository. The labels folder has the data split into the Aachen partition as was done in TrOCR.

The source folder includes the extended HuggingFace TrOCRProcessor and VisionEncoderDecoder so that the the positional embeddings and image sizes can be changed
relatively easily.

The test folder holds the results and code for running various tests on the super computer according to the details in the bash scripts.

The TestHelpers folder holds the main boilerplate code for training and evaluating the models with various parameters that can be used as flags to change what is
printed out and how the model is changed.
