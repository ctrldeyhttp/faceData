from django.db import models
from pgvector.django import VectorField

class UploadedImage(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/')
    date_uploaded = models.DateTimeField(auto_now_add=True)
    embedding = VectorField(dimensions=512, null=True, blank=True) 

    def __str__(self):
        return self.name
