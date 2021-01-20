## MBTI Personality Type Predictor

<img src="https://cdn.80000hours.org/wp-content/uploads/2012/07/home-16personalities.png" alt="drawing" width="300"/>

#### Overview:
The system is a personality predictor based on the MBTI personality types: given a sentence or text input by the user (ideally associated to their values, beliefs, or simply related to them somehow) the system will predict to which personality type the text is mostly related with.

#### Code Execution

1) **set-up your path** : cd to `team03\web-app\mbti_web` 
2) **run the server** : run the command `python manage.py runserver`
3) **use the website** : open your browser and go to `http://localhost:8000/`

##### Support Packages
_Model_
* tensorflow
* keras
* joblib
* pickle

_Pipeline Validator_
* goodtables
* tableschema
* pprint
* shutil
* os
* warnings
* numpy
* pandas
* pandas_schema

_Database Communicator_
* sqlite3
* pandas

_Data Cleaner_
* pandas
* re
* pkg_resources