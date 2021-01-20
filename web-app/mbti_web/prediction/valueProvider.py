import base64
import types

import io
from PIL import Image

# Author: Salvatore Spanu Zucca
# The function returns a dictionary containing data of the desired personality type
# The path and filename is adjusted according to the parameter
# eventual IO errors will result in exception
def getTypeInfo(type):
    
    typesList=['INTJ','INTP','ENTJ','ENTP',
           'INFJ','INFP','ENFJ','ENFP',
           'ISTJ','ISFJ','ESTJ','ESFJ',
           'ISTP','ISFP','ESTP','ESFP']
    if(type.upper() in typesList):
        try:
            typeData= dict()
            basePath="types/" + type.upper() + "/"
            typeData['introduction'] = open(basePath + type.lower() + "_introduction.txt", encoding="utf8").read()
            typeData['phase1'] = open(basePath + type.lower() + "_phase1.txt", encoding="utf8").read()
            typeData['phase2'] = open(basePath + type.lower() + "_phase2.txt", encoding="utf8").read()
            typeData['phase3'] = open(basePath + type.lower() + "_phase3.txt", encoding="utf8").read()
            typeData['personalGrowthDescription'] = open(basePath + type.lower() + "_introduction.txt", encoding="utf8").read()
            typeData['person1name'] = open(basePath + type.lower() + "_1.txt", encoding="utf8").read()
            typeData['person1image']=base64.b64encode(open(basePath + type.lower() + "_1.jpg", "rb").read()).decode()
            typeData['person2name'] = open(basePath + type.lower() + "_2.txt", encoding="utf8").read()
            typeData['person2image'] = base64.b64encode(open(basePath + type.lower() + "_2.jpg", "rb").read()).decode()
            typeData['person3name'] = open(basePath + type.lower() + "_3.txt", encoding="utf8").read()
            typeData['person3image'] = base64.b64encode(open(basePath + type.lower() + "_3.jpg", "rb").read()).decode()
            typeData['person4name'] = open(basePath + type.lower() + "_4.txt", encoding="utf8").read()
            typeData['person4image'] = base64.b64encode(open(basePath + type.lower() + "_4.jpg", "rb").read()).decode()
            return typeData
        except IOError as e:
            print(e)
            return 0
