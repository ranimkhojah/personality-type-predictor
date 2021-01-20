import pandas as pd
from .system import dataCleaner, trainer, reproducibility_tracer as tracer , pipelines_validator
from .system import predictor as preD
from django.shortcuts import render
from django.http import HttpResponse
from django.utils.timezone import make_aware
from datetime import datetime
from .models import Post, Dirty_data, Prediction_model, Package, Training_data, mbti_type, evaluation as EVAL
from keras import backend as K
from tensorflow import Graph, Session
import tensorflow as tf
import joblib
import pickle
import calendar
import time
import os
from datetime import datetime


# Author Duy Nguyen Ngoc
# Load the active model in different sessions
activeModel = Prediction_model.objects.get(running=True)
creation = activeModel.creation
creationStr = creation.strftime("%b-%d_%H_%M_%S")
print(creationStr)
try:
    graph0 = Graph()
    with graph0.as_default():
        session0 = Session(graph=graph0)
        with session0.as_default():
            model_load_0 = joblib.load('./prediction/system/data/model/' + creationStr + '/0.sav')

    graph1 = Graph()
    with graph1.as_default():
        session1 = Session(graph=graph1)
        with session1.as_default():
            model_load_1 = joblib.load('./prediction/system/data/model/' + creationStr + '/1.sav')

    graph2 = Graph()
    with graph2.as_default():
        session2 = Session(graph=graph2)
        with session2.as_default():
            model_load_2 = joblib.load('./prediction/system/data/model/' + creationStr + '/2.sav')

    graph3 = Graph()
    with graph3.as_default():
        session3 = Session(graph=graph3)
        with session3.as_default():
            model_load_3 = joblib.load('./prediction/system/data/model/' + creationStr + '/3.sav')
        

    with open('./prediction/system/data/model/' + creationStr + '/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

except Exception as e:
    print("This server is not up to date and should train befor attempting anything else")






from PIL import Image
from . import valueProvider

#Duy Nguyen Ngoc
def findActiveModel():
    for model in Prediction_model.objects.all():
            if model.running == True:
                return model


#Duy Nguyen Ngoc
#Reload model upon having selected one in the admin view
def reloadModel():
    activeModel = Prediction_model.objects.get(running=True)
    creation = activeModel.creation
    creationStr = creation.strftime("%b-%d_%H_%M_%S")
    print(creationStr)

    global graph0
    global graph1
    global graph2
    global graph3
    global session0
    global session1
    global session2
    global session3
    global tokenizer
    global model_load_0
    global model_load_1
    global model_load_2
    global model_load_3
    session0.close()
    session1.close()
    session2.close()
    session3.close()

    tf.compat.v1.reset_default_graph()

    newGraph0 = Graph()
    with newGraph0.as_default():
        newSession0 = Session(graph=newGraph0)
        with newSession0.as_default():
            model_load_0 = joblib.load('./prediction/system/data/model/' + creationStr + '/0.sav')

    newGraph1 = Graph()
    with newGraph1.as_default():
        newSession1 = Session(graph=newGraph1)
        with newSession1.as_default():
            model_load_1 = joblib.load('./prediction/system/data/model/' + creationStr + '/1.sav')

    newGraph2 = Graph()
    with newGraph2.as_default():
        newSession2 = Session(graph=newGraph2)
        with newSession2.as_default():
            model_load_2 = joblib.load('./prediction/system/data/model/' + creationStr + '/2.sav')

    newGraph3 = Graph()
    with newGraph3.as_default():
        newSession3 = Session(graph=newGraph3)
        with newSession3.as_default():
            model_load_3 = joblib.load('./prediction/system/data/model/' + creationStr + '/3.sav')
           
    with open('./prediction/system/data/model/' + creationStr + '/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


    graph0 = newGraph0
    graph1 = newGraph1
    graph2 = newGraph2
    graph3 = newGraph3
    session0 = newSession0
    session1 = newSession1
    session2 = newSession2
    session3 = newSession3

        


# Takes the user to the admin homepage
def training(request):
    
    if request.method == 'GET':
        models = Prediction_model.objects.all()
        model = findActiveModel()
        package = Package.objects.all().values("name","version").filter(prediction_model=model)
        old0, old1, old2, old3 = grabEvaluationOfModel(model)
        evalu = [old0, old1, old2, old3]
        context = {
            "package" : package,
            "evaluation" : evalu,
            "models" : models
        }
        return render(request, 'prediction/admin.html', context)

    elif request.method == 'POST':

        try:
            csvfile = request.FILES['myfile']
            df = pd.read_csv(csvfile)
            error = pipelines_validator.valuesValidation(df)
            context = {
            "error" :error
            }
            if len(error) > 0:
                return render(request, 'prediction/admin.html', context)
            
            qs = Training_data.objects.all() 
            q = qs.values('type', 'post')
            
            all_data = pd.DataFrame.from_records(q)
            all_data.rename(columns={'post':'posts'}, inplace = True)
            
            if len(df) >= 30:
                data = dataCleaner.seperat_data(df)
            else:
                return HttpResponse(status=400)
           
            if not all_data.empty:
                all_data.append(data)
                all_data.drop_duplicates(subset="posts", keep="first", inplace=True)
                
            else:
                all_data = data
            
            print(all_data.head())
            error = pipelines_validator.valuesValidation(all_data)
            context = {
            "error" :error
            }
            if len(error) > 0:
                return render(request, 'prediction/admin.html', context)

            naive_datetime = datetime.now()
            now = make_aware(naive_datetime)
            nowStr = datetime.now().strftime("%b-%d_%H_%M_%S")

            if not os.path.isdir('./prediction/system/data/model/' + nowStr):
                os.makedirs('./prediction/system/data/model/' + nowStr)
            
            path = './prediction/system/data/model/' + nowStr + '/'        


            eval0, eval1, eval2, eval3 = trainer.trainAllModel(all_data, path)
            new_Prediction_Model = Prediction_model()
            new_Prediction_Model.creation = now
            new_Prediction_Model.running = False
            new_Prediction_Model.save()

            add_model_evaluation_db(eval0["0"], new_Prediction_Model, "char E")
            add_model_evaluation_db(eval0["1"], new_Prediction_Model, "char I")
            
            add_model_evaluation_db(eval1["0"], new_Prediction_Model, "char S")
            add_model_evaluation_db(eval1["1"], new_Prediction_Model, "char N")

            add_model_evaluation_db(eval2["0"], new_Prediction_Model, "char T")
            add_model_evaluation_db(eval2["1"], new_Prediction_Model, "char F")

            add_model_evaluation_db(eval3["0"], new_Prediction_Model, "char J")
            add_model_evaluation_db(eval3["1"], new_Prediction_Model, "char P")

            packages = tracer.getPackagesDataframe(tracer.getPackagesList())

            for lib in packages.iterrows():
                new_lib = Package(
                    name = lib[1]['name'],
                    version = lib[1]['version'],
                    prediction_model = new_Prediction_Model 
                )

                new_lib.save()
        
            for cr, dr in zip(data.iterrows(), df.iterrows()):
                if not Training_data.objects.all().filter(post=cr[1]['posts'], type = cr[1]["type"]):
                    new_clean_data = Training_data(
                        type = cr[1]['type'],
                        post = cr[1]['posts'],
                        date_posted = datetime.now()
                    )

                    new_clean_data.save()
                    new_clean_data.predictor.add(new_Prediction_Model)
                    new_clean_data.save()

                    new_data = Dirty_data(
                        type=dr[1]['type'],
                        post=dr[1]['posts'],
                        date_posted=datetime.now(),
                        training_data = new_clean_data
                    )

                    new_data.save()
                else:
                    print("This already exsist in the DB and dublicates of the data are not allowed")

            return render(request, 'prediction/admin.html')
        
        except Exception as e:
            print(e)
            return HttpResponse(status = 400)

# Author Hakim El Amri 
def evaluation(request):
    try:
        csvfile = request.FILES['myfile']
        df = pd.read_csv(csvfile)
        error= pipelines_validator.valuesValidation(df)
        context = {
            "error" :error
        }
        if len(error) > 0:
            return render(request, 'prediction/admin.html', context)

        data = dataCleaner.seperat_data(df)
        current_model = Prediction_model.objects.latest("id")
        
        old0, old1, old2, old3 = grabEvaluationOfModel(current_model)
        obj0, obj1, obj2, obj3 = re_evaluate(data)
        old = [old0, old1, old2, old3]
        new = [obj0, obj1, obj2, obj3]
        context = {
            "oldEval": old,
            "newEval": new
        }

        return render(request, "prediction/evaluation.html" , context)

    except Exception as e:
        print(e)
        return HttpResponse(e, status=400)

# Author Duy Nguyen Ngoc
def modelChange(request):
    try:
        models = Prediction_model.objects.all()
        selectedModel = request.POST['selectmodel']
        oldModel = Prediction_model.objects.get(running = True)
        oldModel.running = False
        oldModel.save()
        newModel = Prediction_model.objects.get(id = selectedModel)
        newModel.running = True
        newModel.save()
        package = Package.objects.all().values("name","version").filter(prediction_model=newModel)
        old0, old1, old2, old3 = grabEvaluationOfModel(newModel)
        evalu = [old0, old1, old2, old3]
        context = {
            "package" : package,
            "evaluation" : evalu,
            "models" : models
        }
        print("Reloading ML Model.......Please wait.........zzzzzz...")
        reloadModel()
        return render(request, 'prediction/admin.html', context)

    except Exception as e:
        print(e)
        return HttpResponse(e, status=400)

# Author Duy & Hakim El Amri  
def re_evaluate(data):
   
    try :          
        K.set_session(session0)
        with graph0.as_default():
            obj0 = trainer.evaluateNewModel(data, 0, model_load_0, tokenizer)
            obj0["0"]["character"] = "char E"
            obj0["1"]["character"] = "char I"
            obj0["0"]["f1score"] = obj0["0"].pop("f1-score")
            obj0["1"]["f1score"] = obj0["1"].pop("f1-score")
            obj0.pop('weighted avg')
            obj0.pop('macro avg')
            obj0.pop('accuracy')

        K.clear_session()
        print(obj0)
    except Exception as e:
        print("with model 0 :",e)
    
    try :          
        K.set_session(session1)
        with graph1.as_default():
            obj1 = trainer.evaluateNewModel(data, 1, model_load_1, tokenizer)
            obj1["0"]["character"] = "char S"
            obj1["1"]["character"] = "char N"
            obj1["0"]["f1score"] = obj1["0"].pop("f1-score")
            obj1["1"]["f1score"] = obj1["1"].pop("f1-score")
            obj1.pop('weighted avg')
            obj1.pop('macro avg')
            obj1.pop('accuracy')
        K.clear_session()
        print(obj1)
    except Exception as e:
        print("with model 1 :",e)

    try :          
        K.set_session(session2)
        with graph2.as_default():
            obj2 = trainer.evaluateNewModel(data, 2, model_load_2, tokenizer)
            obj2["0"]["character"] = "char T"
            obj2["1"]["character"] = "char F"
            obj2["0"]["f1score"] = obj2["0"].pop("f1-score")
            obj2["1"]["f1score"] = obj2["1"].pop("f1-score")
            obj2.pop('weighted avg')
            obj2.pop('macro avg')
            obj2.pop('accuracy')
        K.clear_session()
        print(obj2)
    except Exception as e:
        print("with model 2 :", e)
    
    try :          
        K.set_session(session3)
        with graph3.as_default():
            obj3 = trainer.evaluateNewModel(data, 3, model_load_3, tokenizer)
            obj3["0"]["character"] = "char J"
            obj3["1"]["character"] = "char P"
            obj3["0"]["f1score"] = obj3["0"].pop("f1-score")
            obj3["1"]["f1score"] = obj3["1"].pop("f1-score")
            obj3.pop('weighted avg')
            obj3.pop('macro avg')
            obj3.pop('accuracy')
        K.clear_session()
        print(obj3)
    except Exception as e:
        print("with model 3 :", e)

    return obj0, obj1, obj2, obj3

# Author Hakim El Amri
# Takes the user to the main user page where the webb app i located
def prediction(request):
    if request.method == 'GET':

        return render(request, 'prediction/user.html')

    elif request.method == 'POST':
        
        sentence = request.POST['Post']

        # Assuming that the pipeline for getting a output from the System will only need one input
        try:
            # commented out code bellow is not fully implementedyet.
            print("Step 1 In the try catch funtion")
            target = dataCleaner.clean_sentence(sentence)
            print("Step 2 Cleaning finsihed")
            
            response , values = preD.makePrediction(tokenizer, target, graph0, graph1, graph2, graph3,
                session0, session1, session2, session3, model_load_0, model_load_1, model_load_2, model_load_3) # This will change into a singel string latter on.
            
            print(values)
            print(response)
            model = Prediction_model.objects.latest("id")

            if(request.user.is_anonymous):

                post_new = Post(
                    content = sentence,
                    type_predicted = response,
                    date_posted = datetime.now(),
                    predictor = model,

                )
            else:  

                post_new = Post(
                    content = sentence,
                    date_posted = datetime.now(),
                    user = request.user,
                    type_predicted = response,
                    predictor = model,
                )

            try: 
                post_new.save()
                print("saved it")

            except Exception as e:
                print(e)

            print("The content of the post was: ", post_new.content," and the response is: ", response)  
            
            value=valueProvider.getTypeInfo((response))
            context = {
                'mbti_type' : response,
                "introduction" : value['introduction'],

                "value1" : values[0],
                "value2" : values[1],
                "value3" : values[2],
                "value4" : values[3],

                "I_percentage" : int(values[0] * 100),
                "E_percentage" : int((1.0-values[0]) * 100)+1,
                "N_percentage" : int(values[1] * 100),
                "S_percentage" : int((1.0-values[1]) * 100)+1,
                "F_percentage" : int(values[2] * 100),
                "T_percentage" : int((1.0-values[2]) * 100)+1,
                "P_percentage" : int(values[3] * 100),
                "J_percentage" : int((1.0-values[3]) * 100)+1,

                "person1name" : value['person1name'],
                "person1image" : value['person1image'],
                "person2name" : value['person2name'],
                "person2image" : value['person2image'],
                "person3name" : value['person3name'],
                "person3image" : value['person3image'],
                "person4name" : value['person4name'],
                "person4image" : value['person4image'],
            }
            return render(request, 'prediction/prediction.html', context) # need to enter a context variable latter
            
        except Exception as e:
            print(e)
            response = HttpResponse( status = 400)

        return response

def image_mat(type):
    return valueProvider.getTypeInfo(type)['person1image']

# Author Hakim El Amri
# IS NOT IMPLEMENTED YET, will return all the users posts
def prediction_list(request):
    posts = Post.objects.all() 
    context = {
        "posts" : posts
    }
    return render(request, 'prediction/prediction.html', context)

# Author Hakim El Amri
def display_models(request):

    package = Package.objects.all().values("name","version","prediction_model")
    check = 0
    response = ""
    word = ""
    end = False

    for pm in package:

        if check != pm["prediction_model"]:
            response += "<br> &emsp; Model Id: "+ str(pm["prediction_model"]) + "<br>"
            check = pm["prediction_model"]

        response += str("&emsp;&emsp;"+pm["name"]) + " v: " + str(pm["version"]) + "<br>"

        print(pm["prediction_model"], "This is object ", pm["name"])



    return HttpResponse(response)

# Author Hakim El Amri
def display_training_data(request):
    mbtitype = Training_data.objects.all().values("type")
    post = Training_data.objects.all().values("post")
    predictor = Training_data.objects.all().values("predictor")
    print(len(Training_data.objects.all()))
    response = ""
    for t, p, pr in zip(mbtitype, post, predictor):
        word = str(t["type"]) +"<br>"+ str(p["post"])+"<br>"+ str(pr)
        word = str(word)
        word += "<br> <br> <br>"
        response += word

    return HttpResponse(response)

# Author Hakim El Amri
def display_dirty_data(request):
    mbtitype = Dirty_data.objects.all().values("type")
    post = Dirty_data.objects.all().values("post")
    predictor = Dirty_data.objects.all().values("training_data")
    print(len(Dirty_data.objects.all()))
    response = ""
    for t, p, pr in zip(mbtitype, post, predictor):
        word = str(t["type"]) +"<br>"+ str(p["post"])+"<br>"+ str(pr)
        word = str(word)
        word += "<br> <br> <br>"
        response += word

    return HttpResponse(response)

# Author Hakim El Amri
def clean_DB(request):
    
    try:
        remove_duplicates_training_data_from_DB()
        return HttpResponse(status = 200)
    except Exception as e:
        print(e)
        return HttpResponse(status = 400)

# Author Hakim El Amri
def remove_duplicates_training_data_from_DB():

    if len(Training_data.objects.all()) > len(Training_data.objects.all().distinct()):

        qs = Training_data.objects.all()
        q = qs.values("id","type", "post","date_posted", "predictor")
        t_data = pd.DataFrame.from_records(q)
        t_data.drop_duplicates(subset="post", keep="first", inplace=True)

        ds = Dirty_data.objects.all()
        d = ds.values("id", "type", "post", "date_posted")
        d_data = pd.DataFrame.from_records(d)
        d_data.drop_duplicates(subset="post", keep="first", inplace=True)


        total = len(Training_data.objects.all())
        Training_data.objects.all().delete()


        for t, d in zip(t_data.iterrows(), d_data.iterrows()) :
            total -= 1
            t_data_new = Training_data(
                id = t[1]["id"],
                type = t[1]['type'],
                post = t[1]['post'],
                date_posted = t[1]['date_posted'],        
            )
            t_data_new.save()
            t_data_new.predictor.add(t[1]['predictor'])

            d_data_new = Dirty_data(
                id = d[1]["id"],
                type = d[1]['type'],
                post = d[1]['post'],
                date_posted = d[1]['date_posted'],
                training_data = t_data_new
            )

            d_data_new.save()


        print("I removed a toatl of : ", total)
        return

    print("Nothing to remove")
    return

# Author: Salvatore Spanu Zucca
# parameter: String
def addTypeData(request):

    typesList=['INTJ','INTP','ENTJ','ENTP',
            'INFJ','INFP','ENFJ','ENFP',
            'ISTJ','ISFJ','ESTJ','ESFJ',
            'ISTP','ISFP','ESTP','ESFP']
            
    for option in typesList:
        value = valueProvider.getTypeInfo(option)
        print(value['person4image'])
        typeData = mbti_type(
            type = option,
            introduction = value['introduction'],
            phase1 = value['phase1'],
            phase2 = value['phase2'],
            phase3 = value['phase3'],
            personalGrowthDescription = value['personalGrowthDescription'],
            person1name = value['person1name'],
            person1image = value['person1image'],
            person2name = value['person2name'],
            person2image = value['person2image'],
            person3name = value['person3name'],
            person3image = value['person3image'],
            person4name = value['person4name'],
            person4image = value['person4image']
        )
        typeData.save()
    return HttpResponse(Status=200)

# Author Hakim El Amri
def retrain_model_from_db(request):
    qs = Training_data.objects.all()
    q = qs.values('type', 'post')
    all_data = pd.DataFrame.from_records(q)
    all_data.rename(columns={'post':'posts'}, inplace = True)

    acc = trainer.trainAllModel(all_data)      

# Author Hakim El Amri
def add_model_evaluation_db(obj, model, name):
    try:
        new_evaluation = EVAL(
            character = name,
            precision = obj["precision"],
            recall = obj["recall"],
            f1score = obj["f1-score"],
            support = obj["support"],
            predictor = model
            )
        new_evaluation.save()
    except Exception as e:

        print(type(e))
        print(e.args)
        print(e)
        return False

    return True

# Author Hakim El Amri
# Will assume you are giving it two objects that are related to each other when ever you invoke the function
def turn_evaluation_obj_to_dict(obj):
    
    newdict = {
            "character": obj["character"],
            'precision': obj["precision"],
            'recall': obj["recall"],
            'f1score':obj["f1score"],
            'support':obj["support"]
            }

    return newdict

# Author Hakim El Amri
def grabEvaluationOfModel(model):

    evaluationData = EVAL.objects.filter(predictor = model).values()
    evo_list =[]

    for row in evaluationData:
        evo_list.append(turn_evaluation_obj_to_dict(row))

    obj0 = {
        "0": next(item for item in evo_list if item["character"] == "char E"),
        "1": next(item for item in evo_list if item["character"] == "char I")
    }

    obj1 = {
        "0": next(item for item in evo_list if item["character"] == "char S"),
        "1": next(item for item in evo_list if item["character"] == "char N")
    }

    obj2 = {
        "0": next(item for item in evo_list if item["character"] == "char T"),
        "1": next(item for item in evo_list if item["character"] == "char F")
    }

    obj3 = {
        "0": next(item for item in evo_list if item["character"] == "char J"),
        "1": next(item for item in evo_list if item["character"] == "char P")
    }

    return obj0, obj1, obj2, obj3
#Author Ranim Khojah
#load images
def images(request):
    return render(request, 'prediction/prediction.html')