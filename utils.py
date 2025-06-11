from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Flatten, Dense, Reshape, UpSampling2D, Concatenate
from tensorflow.keras import models
import numpy as np

import streamlit as st
import cv2


def inicio():
    st.session_state.menu = "inicio"


def pixelar():
    st.session_state.menu = "pixelar"


def estilo_boton():
  st.markdown(
  """
  <style>
  div.stButton > button {
      border: 3px solid #005f73; /* Borde */
      border-radius: 8px; /* Bordes redondeados */
      padding: 12px 28px;
      transition: transform 0.5s; /* Efecto al pasar el mouse */
  }

  div.stButton > button:hover {
      transform: scale(1.15); /* Agrandar un poco al pasar el mouse */
      background-color: #0078a5; /* Color más oscuro */
  }
  </style>
  """, unsafe_allow_html=True)
  



def background():
  st.markdown("""
  <style>
    /* Ocultar el header */
    header { visibility: hidden; }

    /* Quitar todo el padding superior del contenedor principal */
    .block-container {
      padding-top: 0 !important;
    }
    /* Eliminar margen extra en los <h1> */
    h1 {
      margin-top: 0 !important;
    }

    /* Fondo negro y texto rojo en toda la app */
    body, .stApp {
      background-color: black !important;
      color: red !important;
    }
    /* Aplicar rojo al texto en general */
    html, body, [class*="css"] {
      color: red !important;
    }
    /* Forzar títulos en rojo */
    h1, h2, h3, h4, h5, h6 {
      color: red !important;
    }

    /* Inputs y botones con borde y texto rojo, fondo oscuro */
    input,
    .stTextInput > div > div > input,
    .stButton > button {
      border-color: red !important;
      color: red !important;
      background-color: #111 !important;
    }

    /* Etiquetas, descripciones y captions en rojo */
    label,
    .css-1cpxqw2,
    .css-145kmo2,
    .stImageCaption {
      color: red !important;
    }

    /* Botones de descarga personalizados */
    .stDownloadButton > button {
      color: red !important;
      background-color: #222 !important;
      border: 1px solid red !important;
    }

    /* Eliminar sombras */
    * {
      box-shadow: none !important;
    }
  </style>
  """, unsafe_allow_html=True)
    
    
  # File_Uploader
  st.markdown("""
  <style>
  div[data-testid="stFileUploader"] button {
      background-color: #FF0000;
      color: white;
      border: none;
      border-radius: 5px;
  }

  div[data-testid="stFileUploader"] button:hover {
      color: white !important;
      background-color: #CC0000;
  }

  div[data-testid="stFileUploader"] button:active {
      color: white !important;
      background-color: #990000;
  }

  div[data-testid="stFileUploader"] button:focus {
      color: white !important;
      background-color: #CC0000;
      outline: none;
  }
  </style>
""", unsafe_allow_html=True)


def file_uploader(key):
  return st.file_uploader(
            label="Suelta una imagen (JPG, PNG, JPEG)",
            type=["jpg", "png", "jpeg"],
            help="Cargue una imagen (PNG, JPG o JPEG)",
            key=key
        )



def pixelate_face(img):
  face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier(face_cascade_path)
  
  # Detectar caras
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  
  if len(faces) == 1:
    # Pixelar caras detectadas
    for (x, y, w, h) in faces:
        # extraer ROI de la cara
        face_roi = img[y:y+h, x:x+w]

        # Reducir tamaño
        small = cv2.resize(face_roi, (30, 30), interpolation=cv2.INTER_LINEAR)
        # Volver a escala original con bloques
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return pixelated


  error = st.columns([0.5, 0.5, 1, 1])[2]

  with error:
    st.error('No se detectó un rostro. \n\nAsegúrate de que el rostro sea visible y vuelve a intentarlo con otra imagen.')

  return None


def text(alto, n_spaces, texto):
   spaces = 'ㅤ' * n_spaces

   st.markdown(
    f"""
    <style>
    .custom-caption {{
        color: white;
        font-size: 16px;
        text-align: center;
        margin-top: {alto}px;
        margin-bottom: 20px;
        font-weight: 500;
    }}
    </style>
    <p class="custom-caption">{spaces + texto}</p>
    """,  unsafe_allow_html=True)
   


# GENERADOR   ->   Autoencoder (U‑Net ligero)
def generator(input_shape):
    inp = Input(shape=input_shape)
    
    # Encoder
    x  = Conv2D(64, 3, padding='same', activation='relu') (inp)
    e1 = MaxPool2D() (x)
    x  = Conv2D(128,3,padding='same', activation='relu') (e1)
    e2 = MaxPool2D() (x)    # (48x48x128)
    x  = Conv2D(256, 3, padding='same', activation='relu') (e2)
    e3 = MaxPool2D() (x) 
    
    # Bottleneck
    flat = Flatten() (e3) 
    bott = Dense(512, activation='relu') (flat)
    
    # Decoder
    dimensions = (width//8, height//8, 256)        # Tamaño de reconstrucción base (24x24x256) ← inverso al último MaxPool
    x   = Dense(np.prod(dimensions), activation='relu') (bott)
    x   = Reshape(dimensions) (x)
    
    x   = UpSampling2D() (x)
    x   = Conv2D(256, 3, padding='same' ,activation='relu') (x)        # Convolución de refinamiento
    x   = Concatenate() ([x, e2])        # Skip connection.        (48x48x128) → concatena → (48x48x256) = (48x48x384)
    x   = UpSampling2D() (x)
    x   = Conv2D(128, 3, padding='same', activation='relu') (x)
    x   = Concatenate() ([x, e1])
    x   = UpSampling2D() (x)
    x   = Conv2D(64, 3, padding='same', activation='relu') (x)
    
    out = Conv2D(3, 3, padding='same', activation='sigmoid') (x)
    
    return models.Model(inp, out)


# Definición y carga del modelo
width, height = 192, 192
generador = generator((width, height, 3))
generador.load_weights('sources/GAN_18_Gen.weights.h5')




