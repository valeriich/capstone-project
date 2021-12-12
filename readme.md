<center><h1 align="center">MLZoomcamp 2021 capstone project.</h1></center>

<center><h2 align="center">1. Description of the problem</h2></center>
<p>Assume that we have tomato seeds online shop. Client wants to buy the seeds of some kind of tomato, and he does not know the exact name of the cultivar. But he has some pictures of the wanted tomato. So there is a place for application to classify that images and get a name of the tomato cultivar.

<center><h2 align="center">2. Dataset</h2></center>
<a href="https://www.kaggle.com/olgabelitskaya/tomato-cultivars"> Dataset used is from Kaggle "Tomato Cultivars". It contains 776 color images of 15 tomato cultivars.</p>  

<center><h2 align="center">3. How the solution will be used</h2></center>
Given url of user's image, application returns dictionary with class probabilities. Online shop bot chooses the most confident predictions to suggest for user the name of the tomato cultivar needed. User makes an order to purchase seeds of this tomato cultivar.

<center><h2 align="center">4. Description of the repository</h2></center>

1) `notebook.ipynb` - notebook that covers:

  * Data loading, preparation and exploration,
  * Building dataloaders,
  * Building transfer-learning model for image classification with keras,
  * Setting parameters for model,
  * Model training,
  * Monitoring results of the model,
  * Converting tensorflow model to TFLite format.

2) `train.py` - python script to train the model, convert to tflite format and save it.

3) `test.py` - python script to test application.

4) `lambda-function.py` -  python script with function to run application from within the AWS Labmda console.

5) `Dockerfile` - specifies the commands that builds docker image with application.

6) `model.tflite` - tf.keras model converted into TFLite format.

7) `tomatoes.zip` - archived dataset.

8) `readme.md`.

<center><h2 align="center">5. How to run the project</h2></center>

This project supposed to be deployed on AWS Lambda, but this did not happen :(

The validity of the project can be checked in the following way:
* copy (clone) this repository to your local PC to some directory,
* change path to that directory with command line:
```
cd <your-directory-name>
```
* build the docker image:
```
docker build -t tomato-model .
```
* run the docker image with command:
```
docker run -it --rm -p 8080:8080 tomato-model:latest
```
* run the `test.py` script with the image url you want. 
*(check the local IP and paste it instead 'localhost' if you are on Windows)*
* unzip dataset to subfolder `/tomatoes`, if you want to run `train.py` script, or use `!unzip tomatoes.zip` command in the Jupyter notebook