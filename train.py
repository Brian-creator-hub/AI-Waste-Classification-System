import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


print("Starting improved AI Waste Training...")


dataset_path = "dataset"


categories = [
    "Hazardous",
    "Non-Recyclable",
    "Organic",
    "Recyclable"
]


image_size = 128


data = []
labels = []


print("Loading images...")


for idx, category in enumerate(categories):

    folder = os.path.join(dataset_path, category)

    count = 0

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue


        img = cv2.resize(img, (image_size, image_size))


        data.append(img)
        labels.append(idx)

        count += 1


    print(category, "images:", count)



data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)



labels = to_categorical(
    labels,
    num_classes=len(categories)
)



X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)



print("Training:", len(X_train))
print("Testing:", len(X_test))



# Image augmentation

augment = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)



# MobileNetV2 base model

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128,128,3)
)


base.trainable = False



model = Sequential()


model.add(base)


model.add(
    GlobalAveragePooling2D()
)


model.add(
    Dense(
        128,
        activation="relu"
    )
)


model.add(
    Dropout(0.4)
)


model.add(
    Dense(
        len(categories),
        activation="softmax"
    )
)



model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



model.summary()



print("Training...")


model.fit(
    augment.flow(
        X_train,
        y_train,
        batch_size=32
    ),
    epochs=20,
    validation_data=(X_test, y_test)
)



model.save("waste_classifier_model.h5")


print("DONE!")
print("New MobileNetV2 model saved.")