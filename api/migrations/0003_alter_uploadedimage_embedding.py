# Generated by Django 5.1.3 on 2024-12-08 03:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0002_uploadedimage_embedding"),
    ]

    operations = [
        migrations.AlterField(
            model_name="uploadedimage",
            name="embedding",
            field=models.BinaryField(blank=True, null=True),
        ),
    ]
