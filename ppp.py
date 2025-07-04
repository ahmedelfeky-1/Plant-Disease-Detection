import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import os

# تحميل النموذج
model = load_model("plant_disease_model.h5")

# حجم الصور كما تم التدريب
img_size = 128

# إنشاء labels من أسماء الأصناف (نفس الترتيب اللي اتدرب عليه النموذج)
# علشان تكون متوافقة، هنحمل نفس مجلد التدريب مؤقتًا ونستخرج الأصناف
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = r"C:\Users\elfek\Downloads\archive (7)\PlantVillage\PlantVillage"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# توليد أسماء الفئات
labels = list(train_generator.class_indices.keys())

# الكاميرا
cap = cv2.VideoCapture(0)

print("📷 اضغط على حرف 's' لأخذ صورة وتحليلها، واضغط 'q' للخروج.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("كاميرا اللابتوب", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        # تجهيز الصورة
        img = cv2.resize(frame, (img_size, img_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # التنبؤ
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)

        # طباعة معلومات للتصحيح لو حبيت
        print("Index:", class_index)
        print("Labels count:", len(labels))

        # التأكد من أن الفهرس صحيح
        if class_index < len(labels):
            result = labels[class_index]
            confidence = np.max(prediction)
            print(f"🔍 التشخيص: {result} - الدقة: {confidence:.2f}")

            # عرض النتيجة على الصورة
            text = f"{result} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            print("⚠️ فهرس غير صحيح، تأكد من توافق labels مع النموذج.")

        cv2.imshow("النتيجة", frame)
        cv2.waitKey(0)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
