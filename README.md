# Facial-Filters-Using-CNNs
## A CNN based implementation of Instagram-like animal filters, trained using kaggle's Facial Keypoints Detection dataset. Uses OpenCV to generate a real-time video feed! 😺🚀

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
<div align="center">
<img src="https://github.com/probablyabdullah/Facial-Filters-Using-CNNs/assets/79295754/ed418f24-796f-4f28-90d6-860a63263eda" height = "300" width="580"></img>


The Facial Filters Application is an interactive Python program that leverages computer vision and deep learning techniques to apply dynamic animal filters to a person's face in real-time through a webcam feed. This engaging application combines the power of Convolutional Neural Networks (CNNs) for facial keypoints detection with OpenCV for face detection and image manipulation.


<h3>
  🌐<a href="https://www.kaggle.com/c/facial-keypoints-detection/data?select=training.zip" target='_blank'>Link to the kaggle dataset!</a>🌐
</h3>
</div>

## Libraries Used:📂

* [Pandas](https://pandas.pydata.org/) 🔗
* [NumPy](https://numpy.org/) 🔗
* [Scikit-Learn](https://scikit-learn.org/) 🔗
* [TensorFlow](https://www.tensorflow.org/) 🔗
* [Matplot](https://matplotlib.org/) 🔗

## Steps Involved:🛠️

* Download required files into a working directory🛠️<br>
* Download dataset from kaggle🛠️<br>
* Install dependencies from the file🛠️<br>
* Customise code to your specification🛠️<br>
* Train model🛠️<br>
* Run implementation code🛠️<br>

## Steps to implement:✅

* Download this whole repository into a working directory that you created.🧩
* Extract the images.zip file into a images folder.🧩
* Run `pip install -r requirements.txt` in a terminal open in the working directory or in your IDE's integrated terminal. Just make sure the terminal is active inside the working directory.🧩
* Download and the extract the csv file from the kaggle link given above into your working directory.🧩
* Now open the <b>prepare_plot_train_save.ipynb</b> file.🧩
* In the code, at the end, in the `os.chdir` function to save the trained model, change the path of the working directory to the path of your working directory on your local system. <b> Failure in doing so will lead to an error!</b>🧩
* Run the whole jupyter notebook and it should train succesfully for 100 epochs (or however you modify it) and produce and save a <b>model.h5</b> file.🧩
* In case you run into any dependency error, just run  `pip install *package name of missing dependency*`. You can google the name of the packages that we've imported to find out the official names of the packages that you need to type in. Some included in this code are, `opencv-python` for cv2, `scikit-learn` for sklearn, `matplotlib` for matplotlib etc.🧩
* <b>Remember, if any download is in the form of a `.zip` file, you have to extract it's contents to the working directory.</b>🧩
* <b>In case you get an `SyntaxError: (unicode error) 'unicodeescape' codec` error, replace all the `\`(forward slash) with `\\`(double forward slashes) in the path that you specify inside the `os.chdir` command above.</b>🧩
* Run the <b>apply_animal_filters.py</b> file after the model is trained. Run this file either in your IDE's integrated terminal, or in a terminal open in the working directory. Once again, make sure the terminal is active inside the working directory.🧩
* And there you go! Filters on your face!🧩

<hr>
<p align="center">
  <b>
⭐️Developed in collaboration with <a href="https://github.com/shubvrm"> Shubhika Verma </a> for PESU I/O Slot 16. Shubhika Verma is the Subject Matter Expert for the course 'Computer Vision: Introduction to CNNs and YoloV5.'⭐️
  </b>
</p>
