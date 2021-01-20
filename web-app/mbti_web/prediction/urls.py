from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    path('admin/', views.training, name="MBTI Admin"),
    path('', views.prediction, name='MBTI Predict'),
    path('prediction', views.prediction, name='MBTI Result'),
    path('db/training/', views.display_training_data),
    path('db/dirty/', views.display_dirty_data),
    path('db/models/', views.display_models),
    path('admin/wipe/', views.clean_DB),
    path('admin/retrain/', views.retrain_model_from_db),
    path('admin/evaluate/', views.evaluation, name="Admin Evaluate"),
    path('admin/add/type', views.addTypeData),
    path('admin/change', views.modelChange, name = "Change Model"),
]
