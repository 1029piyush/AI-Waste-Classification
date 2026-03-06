import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️")

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("♻️ AI Smart Waste Classification System")
st.write("Upload an image of waste to identify its type and disposal method.")

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = tf.keras.models.load_model(
r"C:\Users\user\OneDrive\Desktop\Waste_Classifier (2).keras"
)

# --------------------------------------------------
# Dataset Classes
# --------------------------------------------------

dataset_classes = [
"Battery","Keyboard","Microwave","Mobile","Mouse",
"PCB","Player","Printer","Television","Washing Machine",
"cardboard","glass","metal","organic","paper","plastic","trash"
]

# --------------------------------------------------
# Category Mapping
# --------------------------------------------------

waste_category_map = {

"Battery":"E-Waste",
"Keyboard":"E-Waste",
"Microwave":"E-Waste",
"Mobile":"E-Waste",
"Mouse":"E-Waste",
"PCB":"E-Waste",
"Player":"E-Waste",
"Printer":"E-Waste",
"Television":"E-Waste",
"Washing Machine":"E-Waste",

"plastic":"Plastic",

"organic":"Biodegradable",
"paper":"Biodegradable",
"cardboard":"Biodegradable",

"glass":"Recyclable",
"metal":"Recyclable",
"trash":"General Waste"
}

# --------------------------------------------------
# Waste Information
# --------------------------------------------------

waste_info = {

"Plastic":{
"Recyclable":"Yes",
"Biodegradable":"No",
"Suggestion":"Send plastic waste to recycling facility."
},

"E-Waste":{
"Recyclable":"Yes",
"Biodegradable":"No",
"Suggestion":"Dispose at an authorized electronic waste recycling center."
},

"Biodegradable":{
"Recyclable":"No",
"Biodegradable":"Yes",
"Suggestion":"Dispose in compost or organic waste bin."
},

"Recyclable":{
"Recyclable":"Yes",
"Biodegradable":"No",
"Suggestion":"Send this material to a recycling plant."
},

"General Waste":{
"Recyclable":"No",
"Biodegradable":"No",
"Suggestion":"Dispose in general waste bin."
}
}

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image):

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    return img

# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
"Upload Waste Image",
type=["jpg","jpeg","png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=350)

    img = preprocess_image(image)

    prediction = model.predict(img)

    predicted_index = int(np.argmax(prediction))

    detected_object = dataset_classes[predicted_index]

    waste_type = waste_category_map[detected_object]

    confidence = prediction[0][predicted_index] * 100

    recyclable = waste_info[waste_type]["Recyclable"]

    biodegradable = waste_info[waste_type]["Biodegradable"]

    suggestion = waste_info[waste_type]["Suggestion"]

    st.markdown("---")

    st.subheader("Detection Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detected Object", detected_object)

    with col2:
        st.metric("Waste Category", waste_type)

    st.metric("Confidence", f"{confidence:.2f}%")

    st.markdown("---")

    st.subheader("Waste Properties")

    col3, col4 = st.columns(2)

    with col3:
        st.info(f"♻ Recyclable: {recyclable}")

    with col4:
        st.success(f"🌱 Biodegradable: {biodegradable}")

    st.markdown("---")

    st.subheader("Disposal Suggestion")

    st.warning(suggestion)