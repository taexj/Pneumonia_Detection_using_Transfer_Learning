import streamlit as st
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained model
model = load_model('model.h5')

def predict_pneumonia(img):
    """Predict whether an image is of a person with pneumonia."""
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    prediction = model.predict(img_preprocessed)
    
    return prediction

def main():
    st.title("Pneumonia Prediction with VGG16")

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption='Uploaded X-ray.', use_column_width=True)

        # Make prediction
        prediction = predict_pneumonia(img)

        # Display prediction
        if prediction[0][0] > prediction[0][1]:
            st.write('Prediction: **Person is safe.**')
        else:
            st.write('Prediction: **Person is affected with Pneumonia.**')
        
        # Optionally display the raw prediction values
        st.write(f'Raw predictions: {prediction}')

if __name__ == '__main__':
    main()
