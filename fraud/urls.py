from django.urls import path
from .views import analyze_fraud

urlpatterns = [
    path("predict/", analyze_fraud),
]