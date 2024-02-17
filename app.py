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
    
    # Improve layout with columns for upload and image display side by side
    col1, col2 = st.columns(2)

    with col1:
        # File uploader allows user to add their own image
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the image in the second column
        with col2:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            st.image(img, caption='Uploaded X-ray.', use_column_width=True)

        # Show a placeholder while computing prediction
        with st.spinner('Analyzing the image...'):
            prediction = predict_pneumonia(img)

            # Display prediction with some styling
            st.markdown("## Prediction Result:")
            if prediction[0][0] > prediction[0][1]:
                st.success('**Person is safe.**')
            else:
                st.error('**Person is affected with Pneumonia.**')
            
            # Optionally display the raw prediction values
            st.write(f'Raw predictions: {prediction}')

if __name__ == '__main__':
    main()
