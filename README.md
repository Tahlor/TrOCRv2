# TrOCR

The IAM folder holds the images from the IAM dataset. The smaller subset of images in the "image" directory was taken directly from the unilm IAM test set that is available for download. The larger set of images in the "AllImages" folder followed the GitHub repository link in the paper, obtained the IAM dataset from the source, and went through the processing steps listed in that GitHub repository. The labels folder has the data split into the Aachen partition as was done in TrOCR.

The source folder includes the extended HuggingFace TrOCRProcessor and VisionEncoderDecoder so that the the positional embeddings and image sizes can be changed
relatively easily. The TestHelpers folder (within the src directory) holds the main boilerplate code for training and evaluating the models with various parameters that can be used as flags to change what is printed out and how the model is changed.

The test folder holds the results and configuration files for running various tests on the super computer according to the details in the config files.

Most of the dependencies should be in the environment.yaml folder, although there were a few packages that conda didn't have, so pip was used. As a result, there is
a requirements.txt file too.

The sync scripts can be used to transfer models from the repository on the supercomputer to your local machine and vice versa. Git can't hold the larger model files, so these scripts can help you maintain the repository in sync. These scripts will completely change the syncing to repository to the syncing from repository, so use them carefully.

# Test Results
To run tests you type ./RunSupercomputerTest.sh followed by the config file path/name like this: ./RunSupercomputerTest.sh FineTune/FineTuneConfig.yml
This is where I ended at:

I wasn't able to avoid the loss spike when using the microsoft/handwritten-small model and training on the IAM images.
I was able to fine tune the stage1 model on the IAM images and avoid the loss spike

Training the learned embeddings only for images with twice the height did not give good model performance
Training the model while using Sinusoidal embeddings (unfreezing and entire model) overfit the training data too quickly before generalizing well to the other images

I didn't quite have the time to work out all of the bugs with the character tokenizer, because occassionally it produces invalid characters and crashes the script
The BERT-base-cased tokenizer was a bit slower than the regular one, so training took longer. I don't think I trained it for long enough to see great results

I tried training a model using double height images with sinusoidal embeddings, but, again, the model overfit the training data before generalizing to the validation data

I tried multiple different types of learning rate schedulers, and found that you shouldn't make the LR warmup too slowly if you want training to occur in a relatively quick time (haha). I didn't see a noticeable difference in performance between when using them or not when they decent numbers of steps, but with more training data it would likely become more apparent

I tested multiple learning rates and found that 5e-6 worked relatively well when training. I experimented with smaller LRs by orders of magnitude down to 5e-8.