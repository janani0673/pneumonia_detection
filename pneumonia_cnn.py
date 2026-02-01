import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# =========================
# Suppress TensorFlow logs
# =========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =========================
# DATASET BASE PATH
# =========================
BASE_DIR = r"D:\pneumonia_detection-main\pneumonia_detection-main\dataset\chest_xray\chest_xray"

train_dir = os.path.join(BASE_DIR, "train")
test_dir  = os.path.join(BASE_DIR, "test")
val_dir   = os.path.join(BASE_DIR, "val")

model_file = "pneumonia_model.h5"

# =========================
# IMAGE DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# =========================
# DATA GENERATORS
# =========================
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

val_generator = None
if os.path.exists(val_dir):
    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )

# =========================
# CNN MODEL
# =========================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# MODEL CHECKPOINT
# =========================
checkpoint = ModelCheckpoint(
    model_file,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =========================
# TRAIN OR LOAD MODEL
# =========================
if not os.path.exists(model_file):
    print("\n🚀 Training model, please wait...\n")
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator if val_generator else test_generator,
        callbacks=[checkpoint]
    )
else:
    print("\n✅ Loaded existing trained model\n")
    model = load_model(model_file)

# =========================
# MODEL EVALUATION
# =========================
test_loss, test_acc = model.evaluate(test_generator)
print("\n✅ Model Evaluation Completed")
print(f"📊 Test Accuracy: {test_acc * 100:.2f}%")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"\n❌ File not found: {img_path}")
        return

    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    print("\n🩻 Prediction Result")
    if prediction > 0.5:
        print("🦠 PNEUMONIA DETECTED")
        print(f"🧠 Confidence: {prediction * 100:.2f}%")
        print("\n💊 Suggested Diagnostic Steps:")
        print("1. Consult a Pulmonologist immediately.")
        print("2. Chest CT scan or additional X-ray may be recommended.")
        print("3. Take prescribed medication as advised.")
        print("4. Maintain proper hydration and rest.")
    else:
        print("💚 NORMAL (No Pneumonia Detected)")
        print(f"🧠 Confidence: {(1 - prediction) * 100:.2f}%")
        print("\n✅ Suggested Follow-up Steps:")
        print("1. Monitor health and temperature.")
        print("2. Recheck if symptoms persist.")
        print("3. Maintain a healthy lifestyle.")

# =========================
# USER INPUT
# =========================
print("\n📸 Enter the full path of the X-ray image to diagnose:")
image_path = input("➡️  ").strip()

predict_image(image_path)
