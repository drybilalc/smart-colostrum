import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.set_page_config(page_title="SMART COLOSTRUM", page_icon="🥛")
st.title("SMART COLOSTRUM 🥛")
st.write("Kolostrum kalitesini analiz etmek için tüpün fotoğrafını çekin.")

@st.cache_resource
def load_model_and_labels():
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
with open("labels.txt", "r") as f:
labels = f.readlines()
return interpreter, labels

try:
interpreter, labels = load_model_and_labels()
except Exception as e:
st.error("Model dosyaları bulunamadı. Lütfen model.tflite ve labels.txt dosyalarının yüklü olduğundan emin ol.")
st.stop()

img_file = st.camera_input("Kolostrum tüpünü kameraya gösterin")

if img_file is not None:
image = Image.open(img_file).convert('RGB')
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)
