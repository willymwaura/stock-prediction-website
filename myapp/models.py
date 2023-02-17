
from django.db import models
from django.utils.timezone import now

# Create your models here.
class Data (models.Model):
    csv_name=models.CharField(max_length=100)
    csv_file=models.FileField(upload_to="csv",blank=False)

class Predicted(models.Model):
    csv_name=models.CharField(max_length=100)
    csv_file=models.FileField(upload_to="pdf",blank=False)

class Ids(models.Model):
    doc_id=models.IntegerField()



