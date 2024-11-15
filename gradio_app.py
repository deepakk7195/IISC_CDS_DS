import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model("glaucoma_model.h5")

# Define a function to preprocess and classify the image
def classify_image(image):
    # Resize the image to the model's input size (assuming 224x224 for this example)
    image = image.resize((224, 224))
    
    # Convert the image to an array and preprocess it
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Get the model prediction
    prediction = model.predict(image_array)
    
    # Assuming the model returns a probability score, with 0 = normal and 1 = glaucoma
    class_label = "Glaucoma" if prediction[0][0] > 0.5 else "Normal"
    
    # Output the result
    return class_label

# Set up the Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.inputs.Image(type="pil"),  # Accept PIL image for preprocessing
    outputs="text",
    title="Fundus Image Glaucoma Classifier",
    description="Upload a fundus image to classify it as Normal or Glaucoma."
)

# Launch the app
demo.launch(share=True)
