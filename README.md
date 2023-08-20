# BrainTumorAI

A CNN (Convoluted Neural Network) AI model that predicts brain tumors based on MRI brain scans.

# Synopsis

Brain tumors represent a serious and often life-threatening health condition, with their severity varying from benign and manageable growths to highly malignant and aggressive forms. The location, size, and rate of growth of the tumor can significantly affect the patient's neurological function and overall prognosis. Early and accurate diagnosis is therefore pivotal for effective treatment planning and improving patient outcomes.

Artificial Intelligence (AI), particularly Convolutional Neural Networks (CNNs) and deep learning techniques, is increasingly playing a transformative role in the early detection and characterization of brain tumors. These advanced algorithms can automatically analyze complex medical imaging data, such as MRI and CT scans, with remarkable precision and speed.

By identifying subtle patterns and abnormalities in these images that might be missed or take significant time for human experts to evaluate, AI tools can significantly expedite the diagnosis process, reduce human error, and enable more timely and targeted treatment interventions, potentially improving survival rates and the quality of life for patients.

BrainTumorAI was designed with the above in mind.  **Please be aware, however, that this AI model is simple and can detect 4 types of classes: Glioma (most deadly), Meningioma, Pituitary, and No Tumor.  This AI model will, therefore, be retrained consistently and further developed by myself and anyone free to contribute to this codebase to make it better.**

# Requirements

Python 3.  You can check if Python 3 is installed on your system by running

```
python3 --version
```

If a version number prints out on the console, you are good to go.

# Setup

**Clone this repository and navigate to the BrainTumorAI directory after this repo has been cloned.**

```
git clone https://github.com/pavondunbar/BrainTumorAI && cd BrainTumorAI
```

**Create a Python virtual environment.  Creating a virtual environment is ideal because the BrainTumorAI will be isolated from other Python projects on your system.  By creating this virtual environment and isolating BrainTumorAI, any library conflicts are avoided.**

```
virtualenv BrainTumorAI
```

If the above command does not work, try the command below:

```
python3 -m venv BrainTumorAI
```

If both commands do not work, you may need to install the venv library using pip. You can install the venv by running this command:

```
pip install virtualenv
```

**Once the BrainTumorAI virtual environment is installed, activate it by running this command:**

```
source BrainTumorAI/bin/activate
```

If all goes well, then you should see in the terminal the (BrainTumorAI) activation next to your command line prompt. 

**NOTE: You can deactivate the BrainTumorAI at any time by typing this command:**

```
deactivate
```

# Install libraries

Run the following command below in the virtual environment to install the libraries needed to train and run the BrainTumorAI model.

```
pip install tensorflow numpy Pillow scikit-learn
```

# Train the BrainTumor AI model

Now the fun part begins.

Run the following command below to begin training the AI model using the "training" dataset provided in this repository.

```
python3 braintumor.py
```

This process will take a while, so grab a coffee or take a short break.  The model is iterating over the dataset 10 times (epochs).  Once the model finishes iterating over the dataset and training itself, it will attempt to classify the type of tumor using the **brain-mri-m-1.jpg** file provided in the repository.

If all goes well, the AI model should output a small classification report and conclude that the MRI sample image submitted is a brain scan of an individual with a **Meningioma tumor.**

# Run A Random Sample Test

The AI model has been trained.  At this point there is no need to retrain the model every time a random scan is submitted.  

After the model has been trained, a new file called **model_weights.h5** will be present in the directory.  This file is created after the AI model has completed training.

From this point forward, to run a scan, simply call the **predict-tumor.py** file using the command below:

```
python3 predict-tumor.py
```

The AI will attempt to predict the type of tumor using another random scan, the **p-tumor.jpeg** file. If successful, the AI model should output that the MRI scan provided is that of an individual with a **Pituitary tumor.**

# Submitting Other Scans

If you want to submit another scan, you will have to do two simple things.

1. Save the random scan in the root directory of the repository.
2. Modify the **predict-tumor.py** file by adding the name of your file as described in the image below:

<img width="858" alt="Screenshot 2023-08-20 at 2 49 11 PM" src="https://github.com/pavondunbar/BrainTumorAI/assets/36899956/3dc9164a-cd08-420b-a678-ec701ecdcca3">

3. Run the **predict-tumor.py** file to do a scan of your new random image.

# Conclusion

This AI is still a work in progress and will consistently be updated to make it more robust and powerful.  Contributions are welcome. Please feel free to make changes to this repository to make this AI better for everyone who wishes to use it in either development or production.




