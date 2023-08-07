import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
# Load the pre-trained model
model = tf.keras.models.load_model('flower3.keras')
# Function to test a single image
def test_single_image(image_path):
    img_size = 150
    # Read and preprocess the image
    img = cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    x = np.array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    # Make prediction
    pred = model.predict(x)
    predicted_class = np.argmax(pred, axis=1)[0]
    # Get the flower name using inverse transform
    flower_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]
    predicted_flower_name = flower_names[predicted_class]
    # Display the result
    plt.imshow(img)
    plt.title(f"Predicted Flower: {predicted_flower_name}")
    plt.show()
# Test a single image from the 'test_flower/' directory
# image_path = 'test_flower/2.png'  # Replace with the path to your test image

root=tk.Tk()
root.withdraw()
filepath=filedialog.askopenfilename()
test_single_image(filepath)
