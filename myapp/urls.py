from django.urls import path
from.import views

urlpatterns = [
    path('',views.index,name='index'),
    path('index',views.index,name='index'),
    path('training',views.training,name='training'),
    path('prediction',views.prediction,name='prediction'),
    path('train',views.train_model,name='train'),
    path('team',views.team,name='team'),
    path('display',views.display,name='display'),
    path('model',views.choose_model,name='model'),
    
]
