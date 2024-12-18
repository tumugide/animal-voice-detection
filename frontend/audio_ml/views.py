# audio_app/views.py
from django.shortcuts import render
from .forms import AudioUploadForm
from django.conf import settings
import os
from ml import AnimalAudioDetector

def upload_audio(request):
    prediction = None
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            audio = request.FILES['audio_file']
            file_path = os.path.join(settings.MEDIA_ROOT, audio.name)
            with open(file_path, 'wb+') as destination:
                for chunk in audio.chunks():
                    destination.write(chunk)
            
            detector = AnimalAudioDetector()
            
            detector.max_length = 215
            
            detector.build_model(
                input_shape=(128, detector.max_length, 1),
                num_classes=4
            )
            detector.model.load_weights(os.path.join(settings.BASE_DIR, 'audio_animal.weights.h5'))
            prediction_idx = detector.predict(file_path)
            classes = ['cat', 'dog', 'duck', 'horse']
            prediction = classes[prediction_idx]
    else:
        form = AudioUploadForm()
    return render(request, 'upload.html', {'form': form, 'prediction': prediction})