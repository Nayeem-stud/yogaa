from django.shortcuts import render, redirect
import time
from userapp.models import *
from adminapp.models import *
from mainapp.models import *
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.conf import settings
from django.core.paginator import Paginator
import matplotlib.pyplot as plt
import io
import base64
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
import numpy as np
from tensorflow.keras.models import load_model
from django.contrib import messages
import pandas as pd
import pytz
import matplotlib
from django.core.files.storage import default_storage
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from PIL import UnidentifiedImageError
from django.http import JsonResponse
from django.utils import timezone


# ------------------------------------------------------


# Create your views here.
def user_dashboard(req):
    prediction_count = UserModel.objects.all().count()
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    Feedbacks_users_count = Feedback.objects.all().count()
    all_users_count = UserModel.objects.all().count()

    if user.Last_Login_Time is None:
        IST = pytz.timezone("Asia/Kolkata")
        current_time_ist = datetime.now(IST).time()
        user.Last_Login_Time = current_time_ist
        user.save()
        return redirect("user_dashboard")

    return render(
        req,
        "user/user-dashboard.html",
        {
            "predictions": prediction_count,
            "user_name": user.user_name,
            "feedback_count": Feedbacks_users_count,
            "all_users_count": all_users_count,
        },
    )


def user_profile(req):
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    if req.method == "POST":
        user_name = req.POST.get("username")
        user_age = req.POST.get("age")
        user_phone = req.POST.get("mobile number")
        user_email = req.POST.get("email")
        user_password = req.POST.get("Password")
        user_address = req.POST.get("address")

        # user_img = req.POST.get("userimg")

        user.user_name = user_name
        user.user_age = user_age
        user.user_address = user_address
        user.user_contact = user_phone
        user.user_email = user_email
        user.user_password = user_password

        if len(req.FILES) != 0:
            image = req.FILES["profilepic"]
            user.user_image = image
            user.user_name = user_name
            user.user_age = user_age
            user.user_contact = user_phone
            user.user_email = user_email
            user.user_address = user_address
            user.user_password = user_password
            user.save()
            messages.success(req, "Updated Successfully.")
        else:
            user.user_name = user_name
            user.user_age = user_age
            user.save()
            messages.success(req, "Updated Successfully.")

    context = {"i": user}
    return render(req, "user/user-profile.html", context)


# ----------------------------------------------------------
import os
import base64
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from django.contrib import messages
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from django.core.files.storage import default_storage
from keras.models import load_model
import os
import numpy as np
import cv2
import base64
from django.shortcuts import redirect, render
from django.contrib import messages
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

# Define class mapping for prediction
class_dict = {
    0: "Adho Mukha Svanasana",
    1: "Anjaneyasana",
    2: "Ardha Matsyendrasana",
    3: "Baddha Konasana",
    4: "Bakasana",
    5: "Balasana",
    6: "Halasana",
    7: "Malasana",
    8: "Salamba Bhujangasana",
    9: "Setu Bandha Sarvangasana",
    10: "Urdhva Mukha Svsnssana",
    11: "Utthita Hasta Padangusthasana",
    12: "Virabhadrasana One",
    13: "Virabhadrasana Two",
    14: "Vrksasana",
}


# Preprocessing functions for each model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = preprocess_input(img_array)  # Preprocess input
    img_array = img_array.reshape(1, 224, 224, 3)  # Add batch dimension
    return img_array


# Prediction functions
def predict_image(image_path, model, class_dict):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_dict.get(predicted_class_index, "Unknown")
    return predicted_class_label


# Load model functions
def load_model_vgg16():
    model_path = os.path.join(settings.BASE_DIR, "yoga_posture_dataset/vgg_yoga.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


def load_model_mobilenet():
    model_path = os.path.join(settings.BASE_DIR, "yoga_posture_dataset/mobilenet.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


def load_model_densenet():
    model_path = os.path.join(
        settings.BASE_DIR, "yoga_posture_dataset/densnet_model.h5"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


# Function to get model info based on model type
def get_model_info(model_type):
    if model_type == "Densenet":
        model_info = Densenet_model.objects.latest("S_No")
    elif model_type == "vgg16":
        model_info = Vgg16_model.objects.latest("S_No")
    elif model_type == "Mobilenet":
        model_info = MobileNet_model.objects.latest("S_No")
    else:
        raise ValueError("Select a valid Model")
    return model_info


# Generate segmented image and encode to base64
def generate_segmented_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    segmented_image_path = os.path.join(settings.MEDIA_ROOT, "segmented_image.jpg")
    cv2.imwrite(segmented_image_path, binary_image)

    grayscale_image_path = os.path.join(settings.MEDIA_ROOT, "grayscale_image.jpg")
    cv2.imwrite(grayscale_image_path, gray_image)

    with open(image_path, "rb") as img_file:
        original_image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    with open(segmented_image_path, "rb") as img_file:
        segmented_image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    with open(grayscale_image_path, "rb") as img_file:
        grayscale_image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    return original_image_base64, segmented_image_base64, grayscale_image_base64

from django.core.files.storage import default_storage
from django.contrib import messages
from django.shortcuts import render, redirect

def Classification(req):
    if req.method == "POST" and req.FILES.get("image"):
        try:
            model_type = req.POST.get("model_type")
            uploaded_file = req.FILES["image"]

            temp_image_path = default_storage.save(uploaded_file.name, uploaded_file)
            image_path = default_storage.path(temp_image_path)

            # Load the appropriate model based on the model_type
            if model_type == "Densenet":
                model = load_model_densenet()
            elif model_type == "vgg16":
                model = load_model_vgg16()
            elif model_type == "Mobilenet":
                model = load_model_mobilenet()
            else:
                raise ValueError("Select a valid Model")

            # Predict the result using the model
            predicted_result = predict_image(image_path, model, class_dict)

            # Print the predicted result for debugging
            print(f"Predicted Result: {predicted_result}")

            # Retrieve model info including name and accuracy
            model_info = get_model_info(model_type)
            model_name = model_info.model_name
            model_accuracy = model_info.model_accuracy

            # Print the model name and accuracy for debugging
            print(f"Model Name: {model_name}")
            print(f"Model Accuracy: {model_accuracy}")

            # Generate the required base64 images
            (
                uploaded_image_base64,
                segmented_image_base64,
                grayscale_image_base64,
            ) = generate_segmented_image(image_path)

            # Store results in the session
            req.session["image_path"] = default_storage.url(temp_image_path)
            req.session["predicted_result"] = predicted_result
            req.session["uploaded_image_base64"] = uploaded_image_base64
            req.session["segmented_image_base64"] = segmented_image_base64
            req.session["grayscale_image_base64"] = grayscale_image_base64
            req.session["model_name"] = model_name
            req.session["model_accuracy"] = model_accuracy

            messages.success(req, "Detection Process Completed")
            return redirect("Classification_result")

        except Exception as e:
            messages.error(req, f"An error occurred: {str(e)}")
            print(f"Error occurred in Classification: {str(e)}")
            return redirect("Classification")

    else:
        return render(req, "user/detection.html")



info = {
    "Adho Mukha Svanasana": {
        "notes": "Downward-Facing Dog. Strengthens arms, legs, and core while lengthening the spine and opening the shoulders. Hands should be shoulder-width apart and feet hip-width apart. Press the heels towards the floor, but it's okay if they don't touch the ground. Engage the quadriceps to help lift the heels."
    },
    "Anjaneyasana": {
        "notes": "Low Lunge Pose. Opens the hips and stretches the quads. Strengthens the legs and core. Ensure the front knee is directly over the ankle and the back leg is extended with the top of the foot or the toes on the ground. You can place the hands on the front thigh or reach them overhead for a deeper stretch."
    },
    "Ardha Matsyendrasana": {
        "notes": "Half Lord of the Fishes Pose. Increases spinal flexibility and stimulates digestion. Keep the spine long and twist from the torso, not the shoulders. Ensure both sit bones remain grounded. You can use a block or bolster under the hips for support if needed."
    },
    "Baddha Konasana": {
        "notes": "Bound Angle Pose. Opens the hips and stretches the inner thighs. Sit up tall and press the soles of the feet together. Gently press the knees towards the floor without forcing. You can use your hands to gently guide the knees down."
    },
    "Bakasana": {
        "notes": "Crow Pose. Builds arm strength and improves balance. Engage the core and keep the gaze forward. Distribute weight evenly across the hands and avoid collapsing the shoulders. It's helpful to keep the elbows bent and spread wide for better balance."
    },
    "Balasana": {
        "notes": "Child's Pose. Provides a restful stretch for the back and hips. Rest the forehead on the mat and stretch the arms forward or alongside the body. Breathe deeply to relax. For added comfort, you can place a bolster or blanket under the hips or knees."
    },
    "Halasana": {
        "notes": "Plow Pose. Stretches the shoulders, spine, and hamstrings. Keep the legs straight and feet on the floor or, if needed, support the lower back with the hands. Avoid turning the head to prevent neck strain. Use a blanket under the shoulders for support."
    },
    "Malasana": {
        "notes": "Garland Pose. Opens the hips and stretches the lower back. Keep the feet flat and knees wide, and press the elbows against the inner knees to deepen the stretch. Maintain an upright torso and engage the core for balance."
    },
    "Salamba Bhujangasana": {
        "notes": "Sphinx Pose. Strengthens the spine and opens the chest. Elbows should be under the shoulders and forearms on the ground. Engage the back muscles and keep the hips and legs relaxed. Avoid pushing the chest too far forward; focus on gently lifting the chest."
    },
    "Setu Bandha Sarvangasana": {
        "notes": "Bridge Pose. Strengthens the back, glutes, and legs. Lift the hips towards the ceiling and clasp the hands under the back. Ensure the feet are hip-width apart and press evenly into the ground. Engage the inner thighs and avoid overextending the lower back."
    },
    "Urdhva Mukha Svsnssana": {
        "notes": "Upward-Facing Dog Pose. Strengthens the spine, opens the chest, and stretches the abdomen. Press the tops of the feet and hands into the floor. Lift the chest and thighs off the ground, but keep the legs active and avoid collapsing the lower back."
    },
    "Utthita Hasta Padangusthasana": {
        "notes": "Extended Hand-to-Big-Toe Pose. Improves balance and stretches the hamstrings. Hold the big toe or use a strap around the foot. Keep the standing leg straight and avoid leaning back. Engage the core to maintain balance."
    },
    "Virabhadrasana One": {
        "notes": "Warrior I Pose. Strengthens the legs and opens the hips and chest. Keep the front knee bent and the back leg straight. Align the front heel with the back arch and reach the arms overhead. The hips should face forward, and you can use a slight torso tilt if needed."
    },
    "Virabhadrasana Two": {
        "notes": "Warrior II Pose. Builds strength and stability in the legs. Keep the front knee bent and the back leg straight. Extend the arms parallel to the ground and gaze over the front hand. Ensure the shoulders are relaxed and not hunched."
    },
    "Vrksasana": {
        "notes": "Tree Pose. Improves balance and strengthens the legs. Place one foot on the inner thigh or calf of the standing leg (avoid the knee). Bring the hands to a prayer position at the chest or reach them overhead. Engage the core and focus on a fixed point to maintain balance."
    },
}


def Classification_result(req):
    if req.method == "GET":
        try:
            image_path = req.session.get("image_path", "")
            predicted_result = req.session.get("predicted_result", "")
            uploaded_image_base64 = req.session.get("uploaded_image_base64", "")
            segmented_image_base64 = req.session.get("segmented_image_base64", "")
            grayscale_image_base64 = req.session.get("grayscale_image_base64", "")

            model_name = req.session.get("model_name", "")
            model_accuracy = req.session.get("model_accuracy", "")

            if predicted_result in info:
                notes = info[predicted_result]["notes"]
            else:
                notes = "No specific information found."

            return render(
                req,
                "user/detection-result.html",
                {
                    "predicted_result": predicted_result,
                    "uploaded_image_base64": uploaded_image_base64,
                    "segmented_image_base64": segmented_image_base64,
                    "grayscale_image_base64": grayscale_image_base64,
                    "model_name": model_name,
                    "model_accuracy": model_accuracy,
                    "info": notes,
                },
            )

        except Exception as e:
            messages.error(req, f"An error occurred: {str(e)}")
            return redirect("Classification")
    else:
        return redirect("Classification")


# ----------------------------------------------------------------------------------------------------

from django.core.files.storage import default_storage
from django.core.files.storage import default_storage
from django.contrib import messages
from django.shortcuts import render, redirect

def user_feedback(req):
    id = req.session["user_id"]
    uusser = UserModel.objects.get(user_id=id)
    
    if req.method == "POST":
        rating = req.POST.get("rating")
        review = req.POST.get("review")
        file_upload = req.FILES.get('fileUpload')

        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)
        sentiment = None
        if score["compound"] > 0 and score["compound"] <= 0.5:
            sentiment = "positive"
        elif score["compound"] >= 0.5:
            sentiment = "very positive"
        elif score["compound"] < -0.5:
            sentiment = "negative"
        elif score["compound"] < 0 and score["compound"] >= -0.5:
            sentiment = "very negative"
        else:
            sentiment = "neutral"

        feedback = Feedback.objects.create(
            Rating=rating, Review=review, Sentiment=sentiment, Reviewer=uusser
        )

        if file_upload:
            try:
                file_name = default_storage.save(f'feedback_files/{file_upload.name}', file_upload)
                feedback.file_upload = file_name
                feedback.save()
            except Exception as e:
                messages.error(req, "Error uploading file: " + str(e))

        messages.success(req, "Feedback recorded")
        return redirect("user_feedback")
    return render(req, "user/user-feedback.html")


from django.shortcuts import render
from .models import Feedback

def display_feedbacks(request):
    feedbacks = Feedback.objects.all().order_by('-datetime')
    return render(request, 'main/b-log.html', {'feedbacks': feedbacks})

def user_logout(req):
    if "user_id" in req.session:
        view_id = req.session["user_id"]
        try:
            user = UserModel.objects.get(user_id=view_id)
            user.Last_Login_Time = timezone.now().time()
            user.Last_Login_Date = timezone.now().date()
            user.save()
            messages.info(req, "You are logged out.")
        except UserModel.DoesNotExist:
            pass
    req.session.flush()
    return redirect("user_login")


# --------------------------------------------------------------------------


def yoga_search(req):
    return render(req, "user/yoga-search.html")

# import cv2
# import numpy as np
# import base64
# from django.http import StreamingHttpResponse, JsonResponse
# from django.shortcuts import render
# from django.conf import settings
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input

# # Load model
# model_path = settings.BASE_DIR / "yoga_posture_dataset/vgg_yoga.h5"
# model = load_model(model_path)

# # Class dictionary
# class_dict = {
#     0: "Adho Mukha Svanasana",
#     1: "Anjaneyasana",
#     2: "Ardha Matsyendrasana",
#     3: "Baddha Konasana",
#     4: "Bakasana",
#     5: "Balasana",
#     6: "Halasana",
#     7: "Malasana",
#     8: "Salamba Bhujangasana",
#     9: "Setu Bandha Sarvangasana",
#     10: "Urdhva Mukha Svsnssana",
#     11: "Utthita Hasta Padangusthasana",
#     12: "Virabhadrasana One",
#     13: "Virabhadrasana Two",
#     14: "Vrksasana",
# }

# # Function to preprocess the frame
# def preprocess_frame(frame):
#     img = cv2.resize(frame, (224, 224))
#     img_array = image.img_to_array(img)
#     img_array = preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Function to detect yoga posture in real-time
# def detect_pose(frame):
#     processed_frame = preprocess_frame(frame)
#     prediction = model.predict(processed_frame)
#     predicted_class_index = np.argmax(prediction)
#     return class_dict.get(predicted_class_index, "Unknown")

# # Video streaming generator
# def video_stream():
#     cap = cv2.VideoCapture(0)  # Open webcam
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect yoga posture
#         pose = detect_pose(frame)

#         # Display detected posture
#         cv2.putText(frame, f"Pose: {pose}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Convert frame to JPEG format
#         _, jpeg = cv2.imencode('.jpg', frame)
#         frame_bytes = jpeg.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

#     cap.release()

# # Streaming response for video feed
# def video_feed(request):
#     return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

# # Render the live detection page
# def live_detection(request):
#     return render(request, "live_detection.html")





# views.py (Django)
import os
import base64
import numpy as np
import cv2
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Global model variable to load once
model = None

def load_yoga_model():
    global model
    if model is None:
        model_path = os.path.join(settings.BASE_DIR, "yoga_posture_dataset/densnet_model.h5")
        model = load_model(model_path)

# Live detection view
def live_detection(request):
    return render(request, 'user/live-detection.html')

# Frame prediction API
def predict_frame(request):
    global model
    if request.method == "POST":
        try:
            # Get base64 image data
            image_data = request.POST.get('image', '')
            if not image_data:
                return JsonResponse({'error': 'No image data received'})

            # Decode base64 image
            header, data = image_data.split(';base64,')
            decoded_data = base64.b64decode(data)
            nparr = np.frombuffer(decoded_data, np.uint8)
            
            # Convert to RGB format
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            img = cv2.resize(img, (224, 224))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Load model if not loaded
            load_yoga_model()

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            yoga_pose = class_dict.get(predicted_class, "Unknown")

            return JsonResponse({
                'prediction': yoga_pose,
                'confidence': float(np.max(predictions))
            })

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})





import re
import requests
from django.conf import settings
from django.shortcuts import render, redirect
from .models import Conversation
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def user_chatbot(request):
    conversations = Conversation.objects.all().order_by('created_at')
    
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if user_message:
            # Call Perplexity API for yoga-related queries
            headers = {
                "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Adjust the prompt to focus on yoga recommendations
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "Act as a friendly yoga assistant. Provide simple and direct responses without numbers or bold formatting."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                headers=headers
            )
            
            bot_response = "Error: Could not get response from AI"
            if response.status_code == 200:
                try:
                    bot_response = response.json()['choices'][0]['message']['content']
                    
                    # Remove markdown bold and any references
                    bot_response = re.sub(r'\\([^]+)\\*', r'\1', bot_response)
                    bot_response = re.sub(r'\[\d+\]', '', bot_response)
                    
                    # Remove numbers and dots from the start of lines
                    bot_response = re.sub(r'^\d+\.\s', '', bot_response, flags=re.MULTILINE)
                    
                    # If user says hello, respond with a simple hello
                    if user_message.lower() == "hello":
                        bot_response = "Hello!"
                    
                    # Encourage daily yoga practice if not already mentioned
                    if "daily" not in bot_response.lower() and user_message.lower() not in ["hello", "hi"]:
                        bot_response += " Remember, practicing yoga daily can enhance flexibility, strength, and mental clarity."
                    
                except:
                    pass
                
            Conversation.objects.create(
                user_message=user_message,
                bot_response=bot_response
            )
            
            return redirect('chatbot')
    
    return render(request, 'user/chatbot.html', {'conversations': conversations})
