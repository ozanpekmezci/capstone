# Detect

## Capstone Project

The aim of this project at hand is to build a software to detect house numbers on streets. The project was featured in the Deep Learning course of Udacity.

All of the Notebooks did perfectly run [Google Colab](<https://colab.research.google.com/>).

The first step of the project that predicts only one digit is located in the `Digit_Recognition.ipynb` file. The second of part the project with multiple digit detection can be found in the file `multi_digit_recognition.ipynb`. The last step with SVHN dataset is programmed in the files `prepare-svhn.ipynb` and `multi_digit_recognition_svhn.ipynb`. As the names suggest,the former one does the pre-processing and the latter one builds and trains the model.

All of the images that used in the report are in `images` folder and the data that are used for manual testing are under `images/test`. `svhn_model.pb` file contains the frozen model that is exported from the notebook. This repository also contains the relevant `proposal` and `report` files written in markdown.