"""
URL configuration for wildfireproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mainapp import views as mainapp_views
from userapp import views as userapp_views
from adminapp import views as admin_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    #main
    path('admin/', admin.site.urls),
    path('',mainapp_views.index,name='index'),
    path('user-login',mainapp_views.user_login,name='user_login'),
    path('admin-login',mainapp_views.admin_login,name='admin_login'),
    path('about-us',mainapp_views.about_us,name='about_us'),
    path('contact-us',mainapp_views.contact_us,name='contact_us'),
    path('register',mainapp_views.register,name='register'),
    path('otp',mainapp_views.otp,name='otp'),
    path('live-detection/', userapp_views.live_detection, name='live_detection'),
    path('predict_frame/', userapp_views.predict_frame, name='predict_frame'),
    #user
    path('user-dashboard',userapp_views.user_dashboard,name='user_dashboard'),
    path('user-chatbot',userapp_views.user_chatbot,name='chatbot'),
    path('user-profile',userapp_views.user_profile,name='user_profile'),
    path('detection',userapp_views.Classification,name='Classification'),
    path('detection-result',userapp_views.Classification_result,name='Classification_result'),
    path('user-feedback',userapp_views.user_feedback,name='user_feedback'),
    path('user-logout',userapp_views.user_logout,name='user_logout'),
    path('user/search',userapp_views.yoga_search,name='yoga_search'),
    path('user/display/feedbacks',userapp_views.display_feedbacks,name='display_feedbacks'),
    # path("live-detection/", userapp_views.live_detection, name="live_detection"),
    # path("video-feed/", userapp_views.video_feed, name="video_feed"),
    #admin
    path('admin-dashboard',admin_views.admin_dashboard,name='admin_dashboard'),
    path('pending-users',admin_views.pending_users,name='pending_users'),
    path('all-users', admin_views.all_users, name='all_users'),
    path('delete-user/<int:user_id>/', admin_views.delete_user, name='delete_user'),
    path('accept-user/<int:id>', admin_views.accept_user, name = 'accept_user'),
    path('reject-user/<int:id>', admin_views.reject_user, name = 'reject'),
    path('change-status/<int:id>', admin_views.change_status, name = 'change_status'),
    path('adminlogout',admin_views.adminlogout, name='adminlogout'),
    path('admin-feedback',admin_views.admin_feedback,name='admin_feedback'),
    path('sentiment-analysis', admin_views.sentiment_analysis, name = 'sentiment_analysis'),
    path('sentiment-analysis-graph',admin_views.sentiment_analysis_graph,name='sentiment_analysis_graph'),
    path('comparision-graph',admin_views.comparision_graph,name='comparision_graph'),
    path('Densenet',admin_views.Densenet,name='Densenet'),
    path('Densenet-result',admin_views.Densenet_result,name='Densenet_result'),
    path('Inception',admin_views.vgg16,name='vgg16'),
    path('Inception-result',admin_views.Vgg16_result,name='Vgg16_result'),
    path('Mobilenet',admin_views.Mobilenet,name='Mobilenet'),
    path('Mobilenet-result',admin_views.Mobilenet_result,name='Mobilenet_result'),
    path('Train-Test-Split',admin_views.Train_Test_Split,name='Train_Test_Split'),
    path('Train-Test-Split-Result',admin_views.Train_Test_Split_Result,name='Train_Test_Split_Result'),
    
    
    
]+ static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
