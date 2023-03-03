from django.urls import path
from . import views
urlpatterns = [
    path('baseline/', views.main_page, name='main_page'),
    path('baseline/submit/', views.create_task, name='create_task'),
    path('baseline/result/<uuid>/', views.task_result, name='task_result'),
]