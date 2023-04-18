# TrOCR

The IAM folder holds the images from the IAM dataset. The smaller subset of images in the "image" directory was taken directly from the unilm IAM test set that is available for download. The larger set of images in the "AllImages" folder followed the GitHub repository link in the paper, obtained the IAM dataset from the source, and went through the processing steps listed in that GitHub repository. The labels folder has the data split into the Aachen partition as was done in TrOCR.

The source folder includes the extended HuggingFace TrOCRProcessor and VisionEncoderDecoder so that the the positional embeddings and image sizes can be changed
relatively easily. The TestHelpers folder (within the src directory) holds the main boilerplate code for training and evaluating the models with various parameters that can be used as flags to change what is printed out and how the model is changed.

The test folder holds the results and configuration files for running various tests on the super computer according to the details in the config files.

Most of the dependencies should be in the environment.yaml folder, although there were a few packages that conda didn't have, so pip was used. As a result, there is
a requirements.txt file too.

The sync scripts can be used to transfer models from the repository on the supercomputer to your local machine and vice versa. Git can't hold the larger model files, so these scripts can help you maintain the repository in sync. These scripts will completely change the syncing to repository to the syncing from repository, so use them carefully.