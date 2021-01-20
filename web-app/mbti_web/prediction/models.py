from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.contrib.auth.models import User

## By default, Django gives each model the following field:
## id = models.AutoField(primary_key=True)

# Author: Salvatore Spanu Zucca
class Training_data(models.Model):
    # includes id = PK
    type = models.CharField(max_length = 4)
    post = models.TextField()
    date_posted = models.DateTimeField(auto_now_add=True)
    predictor = models.ManyToManyField("Prediction_model")
    objects = models.Manager()

# Author: Salvatore Spanu Zucca
# this model will use django built in validators to keep the model's accuracy in range between 0-100%
# it ideally bundles the 4 inference models (I/E, N/S, T/F, J/P) into one single reference
class Prediction_model(models.Model):
    # includes id = PK
    creation = models.DateTimeField(null=True, blank=True)
    running= models.BooleanField(default=False)
    objects = models.Manager()

# Author: Salvatore Spanu Zucca
class Package(models.Model):
    name = models.TextField(primary_key=True)
    version = models.TextField()
    prediction_model = models.ForeignKey(Prediction_model, on_delete=models.CASCADE)
    objects = models.Manager()

# Author: Salvatore Spanu Zucca
class Post(models.Model):
    # includes id = PK
    content = models.TextField()
    date_posted = models.DateTimeField(auto_now_add=True)
    type_predicted = models.CharField(max_length = 4)
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank = True, null= True)
    predictor = models.ForeignKey(Prediction_model, on_delete = models.SET_NULL, null=True )
    objects = models.Manager()

    def __str__(self):
        return self.type_predicted + "<br>" + self.content + "<br>"

# Author Hakim El Amri
class Dirty_data(models.Model):
    # includes id = PK
    type = models.CharField(max_length = 4)
    post = models.TextField()
    date_posted = models.DateTimeField(auto_now_add=True)
    training_data = models.OneToOneField(Training_data, on_delete=models.CASCADE)
    objects = models.Manager()

    def __str__(self):
        return self.type 

# Author: Salvatore Spanu Zucca
# this model contains description and resources related to user value proposition of each mbti type
# due to technological restrictions (sqlite not supporting array datatypes unlike postgres) this class will not
# receive further modularization
# The number of similar people will be 8 (4 good, 4 bad)
class mbti_type(models.Model):
    type = models.CharField(max_length= 4, primary_key=True)
    introduction = models.TextField()
    person1name = models.TextField()
    person1image = models.ImageField()
    person2name = models.TextField()
    person2image = models.ImageField()
    person3name = models.TextField()
    person3image = models.ImageField()
    person4name = models.TextField()
    person4image = models.ImageField()
    objects = models.Manager()

# Author: Salvatore Spanu Zucca
# this model table replaces the single accuracy attribute originally used in the Prediction model table
# intended to include evaluation measures for all the 4 inference models (one for each cognitive learning style)
class evaluation(models.Model):
    # includes id = PK
    character = models.TextField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1score = models.FloatField()
    support = models.FloatField()
    predictor = models.ForeignKey(Prediction_model, on_delete=models.CASCADE)
    objects = models.Manager()

# Author: Salvatore Spanu Zucca
# removes duplicate entries from the dirty data table
def removeDirtyDataDuplicates():
    for row in Dirty_data.objects.all():
        if Dirty_data.objects.filter(post=row.post).count() > 1:
            row.delete()

# Author: Salvatore Spanu Zucca
# removes duplicate entries from the training data table
def removeDirtyDataDuplicates():
    for row in Training_data.objects.all():
        if Dirty_data.objects.filter(post=row.post).count() > 1:
            row.delete()