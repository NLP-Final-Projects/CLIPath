from django.urls import path
from . import views
urlpatterns = [
    path('', views.main_page, name='main_page'),
    path('submit/', views.create_task, name='create_task'),
    path('result/<uuid>/', views.task_result, name='task_result'),
]