from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
from . import for_lance

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),
    path('success/', views.success, name='success'),
    path('images/', views.show_uploaded_images, name='show_uploaded_images'),
    path('bilislikas/', for_lance.upload_image, name='upload_image'),

    # path('result/', views.compare_face_to_uploaded_images, name='result'),  # This could be the result page if you want a specific URL
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)