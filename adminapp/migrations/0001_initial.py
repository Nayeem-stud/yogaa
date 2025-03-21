# Generated by Django 5.1 on 2024-08-08 20:09

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="All_users_model",
            fields=[
                ("User_id", models.AutoField(primary_key=True, serialize=False)),
                ("user_Profile", models.FileField(upload_to="images/")),
                ("User_Email", models.EmailField(max_length=50)),
                ("User_Status", models.CharField(max_length=10)),
            ],
            options={
                "db_table": "all_users",
            },
        ),
        migrations.CreateModel(
            name="Densenet_model",
            fields=[
                ("S_No", models.AutoField(primary_key=True, serialize=False)),
                ("model_accuracy", models.CharField(max_length=10)),
                ("model_name", models.CharField(max_length=10)),
                ("model_executed", models.CharField(max_length=10, null=True)),
            ],
            options={
                "db_table": "Densenet_model",
            },
        ),
        migrations.CreateModel(
            name="MobileNet_model",
            fields=[
                ("S_No", models.AutoField(primary_key=True, serialize=False)),
                ("model_accuracy", models.CharField(max_length=10)),
                ("model_name", models.CharField(max_length=10)),
                ("model_executed", models.CharField(max_length=10, null=True)),
            ],
            options={
                "db_table": "MobileNet_model",
            },
        ),
        migrations.CreateModel(
            name="Train_test_split_model",
            fields=[
                ("S_No", models.AutoField(primary_key=True, serialize=False)),
                ("Images_training", models.CharField(max_length=10, null=True)),
                ("Images_validation", models.CharField(max_length=10, null=True)),
                ("Images_testing", models.CharField(max_length=10, null=True)),
                ("Images_classes", models.CharField(max_length=10, null=True)),
            ],
            options={
                "db_table": "Traintestsplit",
            },
        ),
        migrations.CreateModel(
            name="Vgg16_model",
            fields=[
                ("S_No", models.AutoField(primary_key=True, serialize=False)),
                ("model_accuracy", models.CharField(max_length=10)),
                ("model_name", models.CharField(max_length=10)),
                ("model_executed", models.CharField(max_length=10, null=True)),
            ],
            options={
                "db_table": "Vgg16_model",
            },
        ),
    ]
