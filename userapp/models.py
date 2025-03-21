from django.db import models
from mainapp.models import *


class Feedback(models.Model):
    Feed_id = models.AutoField(primary_key=True)
    Rating = models.CharField(max_length=100, null=True)
    Review = models.CharField(max_length=225, null=True)
    Sentiment = models.CharField(max_length=100, null=True)
    Reviewer = models.ForeignKey(UserModel, on_delete=models.CASCADE, null=True)
    datetime = models.DateTimeField(auto_now=True)
    file_upload = models.FileField(upload_to='media/', null=True, blank=True)

    class Meta:
        db_table = "feedback_details"

from django.db import models

class PredictionResult(models.Model):
    prediction_id = models.AutoField(primary_key=True)
    predicted_class = models.CharField(max_length=100, null=True)
    prediction_accuracy = models.DecimalField(max_digits=5, decimal_places=2, null=True)
    datetime = models.DateTimeField(auto_now=True)


    class Meta:
        db_table = "prediction_results"



class Conversation(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def _str_(self):
        return f"User: {self.user_message[:50]}..."
