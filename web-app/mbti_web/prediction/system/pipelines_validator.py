import pandas as pd
from goodtables import validate
from tableschema import infer, Schema
from shutil import copyfile
import os
import warnings
import numpy as np
from pandas_schema import Column, Schema as sc
from pandas_schema.validation import IsDistinctValidation, InListValidation, CustomElementValidation


# Author: Salvatore Spanu Zucca and Ranim Khojah
# Creates the json schema to compare new CSV data with
def createValidationSchema(file):
    try:
        if os.path.exists(file):
            descriptor = infer(file)
            schema = Schema(descriptor)
            schema.save('schema-inferred.json')
            with open('schema-inferred.json') as f:
                txt = f.read()
            f.closed
            try:
                if not os.path.isfile('schema.json'):
                    copyfile(src='schema-inferred.json', dst='schema.json')
                    return 1
                else:
                    raise FileExistsError("the file schema.json already exist")
            except FileExistsError as error:
                print(error)
        else:
            raise FileNotFoundError("the file", file, "doesn't exist")
    except FileNotFoundError as err:
        print(err)


#data = 'test_files/clean_mbti_db.csv'
#sch= 'test_files/schema.json'
#createValidationSchema(data)

# Author: Salvatore Spanu Zucca and Ranim Khojah
# the function returns the number of errors occurred during schema comparison between
# data being served and the json schema

def schema_validateServingData(sch, file):
    try:
        if not os.path.exists(sch):
            raise FileNotFoundError("the file", sch , "doesn't exist")
            quit()
        if not os.path.exists(file):
            raise FileNotFoundError("the file", file, "doesn't exist")
            quit()
        else:
            schema = Schema(sch)
            report = validate(file, checks=['schema'], schema=schema.descriptor)

            # print(report)
            numErrors = report['tables'][0]['error-count']
            if(numErrors>0):
                warnings.warn("Error(s) on the serving data scheme validation!")
                # NOTE: USE ['tables'] [index] ['subtable'] for nested list elements
                numErrors = report['tables'][0]['error-count']
                print("Number of errors: ", numErrors)
                print("Error(s) message ",report['tables'][0]['errors'][0]['message'])
            elif (numErrors==0):
                print("NO ERRORS!")
            return numErrors
        
    except (FileExistsError, FileNotFoundError) as error:
        print(error)


def check_not_number(num):
    try:
        int(num)
    except ValueError:
        return True
    try:
        float(num)
    except ValueError:
        return True
    return False

#schema_validateServingData(sch, data)

# Author: Salvatore Spanu Zucca
# Checks the CSV posts to be unique with each other (avoids duplicates)
# mbti types to be among the possible ones
# checks both being NOT NULL
# validates values being NOT INTEGER
# Return 0 : no errors
# else return errors list
def valuesValidation(data):
    types=['INTJ','INTP','ENTJ','ENTP',
           'INFJ','INFP','ENFJ','ENFP',
           'ISTJ','ISFJ','ESTJ','ESFJ',
           'ISTP','ISFP','ESTP','ESFP']
    schema = sc([
        Column('posts', [IsDistinctValidation(),CustomElementValidation(lambda d: d is not np.nan, 'this field cannot be null'), CustomElementValidation(lambda i: check_not_number(i), 'is an integer')]),
        Column('type', [InListValidation(types), CustomElementValidation(lambda d: d is not np.nan, 'this field cannot be null')]),
    ])
    errors = schema.validate(data)
    # errors
    if len(errors) != 0:
        print("Values Validation FAILED!")
    # no errors
    else: print("Values Check OK!")
    return errors


#data = pd.read_csv("mbti-type/mbti_1.csv")
#errors= valuesValidation(data)
#print(len(errors))