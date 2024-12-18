# audio_app/index.py
import numpy as np
import librosa
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class AnimalAudioDetector:
    def __init__(self, sample_rate=22050, duration=5, max_length=None):
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_length = max_length
        self.model = None
    
    def extract_features(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate, duration=self.duration)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram
    
    def prepare_feature(self, feature):
        if self.max_length is None:
            self.max_length = feature.shape[1]
        if feature.shape[1] < self.max_length:
            padded_feature = np.pad(feature, ((0, 0), (0, self.max_length - feature.shape[1])), mode='constant')
        else:
            padded_feature = feature[:, :self.max_length]
        return padded_feature.reshape(1, feature.shape[0], self.max_length, 1)
    
    def build_model(self, input_shape, num_classes):
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model built and compiled.")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size
        )
        print("Training completed.")
        self.model.save_weights('audio_animal.weights.h5')
        return history
    
    def predict(self, audio_path):
        feature = self.extract_features(audio_path)
        prepared_feature = self.prepare_feature(feature)
        prediction = self.model.predict(prepared_feature)
        return np.argmax(prediction)
    
    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return accuracy * 100