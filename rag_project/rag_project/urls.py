from django.urls import path
from .views import chat_view

urlpatterns = [
    path('api/chat/', chat_view, name='chat'),
]
