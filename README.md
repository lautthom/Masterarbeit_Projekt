# Master thesis: Personalized Recurrent Neural Network Models for Pain Recognition

The code is written in and tested for `Python 3.10.6`. The required packages are listed in `requirements.txt`.
The code is best run in a virtual environment.

Before running the code the folder `PartC-Biosignals` containing the Part C of the BioVid Heat Pain Database needs to be copied into the `data/raw` directory.
After that the code can be run. Note, that the code should be run from the root folder. It can be run using:
```
python src/main.py
```
This will run one leave-one-proband-out run using all models under one of the test conditions.
The test conditions can be changed at the bottom of the `main.py` file.
Note, that the preprocessing of the data can take some time, however once the data was preprocessed, it will be saved. By setting `do_preprocessing` at the bottom of the `main.py` file to `False` the preprocessing will be skipped in future training runs.

Running the `main.py` script will create result files in the `results` folder containing the achieved accuracy for each proband.
To create updated files containing total results and results for the age-gender groups the `src/group_results_loso.py` script needs to be run.

The results and models reported in the thesis are provided in the `results` and `models` folders, repectively.
If you want to train new models, the respective models in the `models` folder need to be deleted. The results will be replaced automatically.
Note, that downloading the repository can take some time, due to the models taking up a lot of memory.


