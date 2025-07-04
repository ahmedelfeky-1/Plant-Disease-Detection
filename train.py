import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# إعداد المسار
data_dir = r"C:\Users\elfek\Downloads\archive (7)\PlantVillage\PlantVillage"
img_size = 128
batch_size = 32

# تحميل الصور
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size),
    batch_size=batch_size, subset='training', class_mode='categorical'
)
val = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size),
    batch_size=batch_size, subset='validation', class_mode='categorical'
)

# بناء الموديل
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# التدريب
model.fit(train, epochs=10, validation_data=val)

# حفظ النموذج
model.save("plant_disease_model.h5")
import json

# حفظ أسماء التصنيفات في ملف labels.json
with open("labels.json", "w") as f:
    json.dump(list(train.class_indices.keys()), f)

print("✅ تم حفظ النموذج وملف التصنيفات labels.json")
