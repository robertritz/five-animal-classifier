import streamlit as st
from fastai.vision.all import *
import gdown

st.markdown("""# Mongolian Five Animal Classifier

Mongolia has five traditional herd animals: Horses, Cattle, Sheep, Goats, and Camels. This app can perform image classification on these animals. Upload an image and try it out!""")

st.markdown("""### Upload your image here""")

image_file = st.file_uploader("Upload an Image", type=["png","jpg","jpeg"])

## Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1hks-274Jgo6cWFhijBwTR2WjSczUxNhR'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

if image_file is not None:
    img = PILImage.create(image_file)
    st.image(img)

    pred, pred_idx, probs = learn_inf.predict(img)

    st.markdown(f"""**This is an image of:** {pred}""")
    st.markdown(f"""Confidence: {max(probs.tolist())}""")