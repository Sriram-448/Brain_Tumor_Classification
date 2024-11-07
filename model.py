import streamlit as st
from keras.models import load_model
import tensorflow as tf
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# Load the model 
model = tf.keras.models.load_model(r"C:\Users\budde\OneDrive\Desktop\Vivek\MiniProject\MiniProject\Vivek\model_parameters_1.keras")
# vgg_19=tf.keras.models.load_model(r"C:\Users\budde\OneDrive\Desktop\Vivek\MiniProject\MiniProject\Vivek\model_vgg19.keras")

print("VGG model loaded")
print("VGG19 model loaded")
test=r"C:\Users\budde\OneDrive\Desktop\Vivek\MiniProject\MiniProject\Model\test_data"
train_data_generator = ImageDataGenerator(rescale=1.0/255)
test_data = train_data_generator.flow_from_directory(
    directory=test,
    target_size=(200, 200),
    batch_size=42,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False 
)
print("Model and test data loaded")
st.title("Brain Tumor Classification")

def homePage():
    image_data1 = {
        "Glioma": r"C:\Users\budde\OneDrive\Desktop\Vivek\MiniProject\Braintumors\glioma_tumor\gg (49).jpg",
        "Meningioma": r"C:\Users\budde\OneDrive\Desktop\Vivek\MiniProject\Braintumors\meningioma_tumor\m (18).jpg",
        "No_Tumor": r"C:\Users\budde\OneDrive\Desktop\Vivek\MiniProject\Braintumors\no_tumor\no (420).jpg",
        "Pitutary": r"C:\Users\budde\OneDrive\Desktop\Vivek\MiniProject\Braintumors\pituitary_tumor\p (37).jpg",
    }
    keys1 = list(image_data1.keys())
    st.subheader("Original Images")
    for i in range(0, 4, 2):
        col1, col2= st.columns(2)
        with col1:
            st.image(image_data1[keys1[i]], width=150, caption=keys1[i])
        with col2:
            st.image(image_data1[keys1[i + 1]], width=150, caption=keys1[i + 1])
        st.markdown("---")
    image_data2 = {
        "Glioma": r"test_data\glioma_tumor\gg (49).jpg",
        "Meningioma": r"test_data\meningioma_tumor\m (18).jpg",
        "No_Tumor": r"test_data\no_tumor\no (420).jpg",
        "Pitutary": r"test_data\pituitary_tumor\p (37).jpg",
    }
    keys2 = list(image_data2.keys())
    st.subheader("Resized and Augmented Images")
    for i in range(0, 4, 2):
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_data2[keys2[i]], width=150, caption=keys2[i])
        with col2:
            st.image(image_data2[keys2[i + 1]], width=150, caption=keys2[i + 1])
        st.markdown("---")

def render_option_1():
    st.title("Modified VGGNET model")
    st.write("This is custom content for VGG.")
    #details about thr data
    data = {
    'Data' : ['No of images',"images per of class","Accuracy"],
    'Train': [7124, 1780, 99.8],
    'Test': [2039, 509, 98.479],
    'Validation': [1020,255, 98.4]
        }
    df = pd.DataFrame(data)
    st.table(df)  
    # for testing on the spot
    if st.button('Test Accuracy'):
        st.write("Testing Started")
        predict=model.predict(test_data)
        predict_=np.argmax(predict,axis=1)
        tr=test_data.classes
        st.write(f"model accuracy{accuracy_score(tr,predict_)}")
        #for printing classification report in realtimne
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(tr, predict_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()
        #for printing classification report in realtimne
        st.subheader('Classification Report')
        cr = classification_report(tr, predict_, output_dict=True)
        df_cr = pd.DataFrame(cr).transpose()
        st.write(df_cr)


def render_option_2():
    st.title("VGG 19 model")
    st.write("This is custom content for VGG.")
    #details about thr data
    data = {
    'Data' : ['No of images',"per of data","Accuracy"],
    'Train': [7124, 1780, 93.5],
    'Test': [2039, 509, 92.3],
    'Validation': [1020,255, 91.0]
        }
    df = pd.DataFrame(data)
    st.table(df)
    # for testing on the spot
    if st.button('Test Accuracy'):
        st.write("Testing Started")
        predict=model.predict(test_data)
        predict_=np.argmax(predict,axis=1)
        tr=test_data.classes
        st.write(f"model accuracy{accuracy_score(tr,predict_)}")
        #for printing classification report in realtimne
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(tr, predict_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()
        #for printing classification report in realtimne
        st.subheader('Classification Report')
        cr = classification_report(tr, predict_, output_dict=True)
        df_cr = pd.DataFrame(cr).transpose()
        st.write(df_cr)




def preprocess_image(image):#here preprocessing is done
    img_array = np.array(image)
    normalized_image = img_array / 255.0  
    return normalized_image
#predicts vthe class for uploaded image
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Convert file-like object to image
    image = Image.open(uploaded_file)
    # Converting image to numpy array
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Display the grayscale image
    st.image(gray_image, caption='Grayscale Image', width=200)
    resized_image = cv2.resize(gray_image, (200, 200))
    preprocessed_image = preprocess_image(resized_image) 
    if st.button('Perform Inference'):
        # The test for the images starts after clicking
        pred1 = model.predict(np.expand_dims(preprocessed_image, axis=0))
        #pred1 is a probability distribution over the classes
        class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        predicted_class_index = np.argmax(pred1, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        st.write(f"Predicted class: {predicted_class_label}")


           
def side_bar():
    st.sidebar.title("Sidebar with Dropdown for selecting model")
    # A dropdown in the sidebar
    dropdown_options = ["Select","VGGNet","VGG19"]
    selected_option = st.sidebar.selectbox("Select an option", dropdown_options)
    # Render content based on the selected option
    if selected_option == dropdown_options[1]:
        render_option_1()
    elif selected_option == dropdown_options[2]:
        render_option_2()
        pass
    # elif selected_option == dropdown_options[3]:
    #     #render_option_3()
        # pass
    else:
        homePage()
side_bar()