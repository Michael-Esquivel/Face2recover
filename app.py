import streamlit as st
from utils import *
import numpy as np
import cv2


st.set_page_config(layout="wide")
background()
estilo_boton()


if "menu" not in st.session_state:
    st.session_state["menu"] = "inicio"  


# FaceRecover
if st.session_state["menu"] == "inicio":
    logo, space, title, change_menu = st.columns([0.5, 0.5, 1, 1])

    # Logo
    with logo:
        st.image("sources/Logo.png", width=500)

    with title:
        st.image("sources/Despixelizar.png", width=500)
        pixel_files = file_uploader(1)

    if pixel_files:
        # Lectura y preprocesado
        file_bytes = np.asarray(bytearray(pixel_files.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb  = cv2.resize(img_rgb, (width, height))
        img_norm = img_rgb.astype(np.float32) / 255.0

        batch  = np.expand_dims(img_norm, axis=0)
        y_pred = generador.predict(batch)[0]


        sp, img, sp2, img2, _, save_img = st.columns([0.80, 1, 1, 1, 0.3, 0.87])

        with img:
            st.image(img_rgb, caption=".", width=550)
            text(-45, 9, 'Imagen pixelada')

        with img2:
            st.image(y_pred, caption=".", width=550)
            text(-45, 9, 'Imagen Generada')

        with save_img:
                st.write("<br>" * 21, unsafe_allow_html=True)

                img_uint8 = (y_pred * 255).astype(np.uint8)
                img_bgr_to_save = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                
                _, buffer = cv2.imencode('.jpg', img_bgr_to_save)
                orig_name = pixel_files.name.rsplit('.', 1)[0]
                

                st.download_button(
                    label="Guardar",
                    data=buffer.tobytes(),
                    file_name=f"clean_{orig_name.replace('pixel_', '')}.jpg",
                    mime="image/jpeg",
                    key=5
                )

    with change_menu:
        st.write("<br>" * 1, unsafe_allow_html=True)
        st.button("Pixelizar", key=2, on_click=pixelar)
            

# Pixelizador
elif st.session_state["menu"] == "pixelar":
    logo, space, title, change_menu = st.columns([0.5, 0.5, 1, 1])

    # Logo
    with logo:
        st.image("sources/Logo.png", width=500)

    with title:
        st.image("sources/Pixelizar.png", width=400)
        clean_files = file_uploader(55)


    if clean_files:
        # Lectura y preprocesado
        file_bytes = np.asarray(bytearray(clean_files.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pixel_img = pixelate_face(img_rgb)
        if pixel_img is not None:
            _, colA, _, colB, _, save_img = st.columns([0.80, 1, 1, 1, 0.3, 0.87])

            with colA:
                st.image(img_rgb, caption=".", width=550)
                text(-50, 9, 'Imagen original')

            with colB:
                st.image(pixel_img, caption=".", width=550)
                text(-50, 9, 'Imagen pixelada')


                with save_img:
                    st.write("<br>" * 21, unsafe_allow_html=True)
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(pixel_img, cv2.COLOR_BGR2RGB))

                    orig_name = clean_files.name.rsplit('.', 1)[0]  # sin extensión

                    st.download_button(
                        label="Guardar",
                        data=buffer.tobytes(),
                        file_name=f"pixel_{orig_name}.jpg",
                        mime="image/jpeg",
                        key=5
                    )

    with change_menu:
        st.write("<br>" * 1, unsafe_allow_html=True)
        st.button("Despixelizar", key=4, on_click=inicio)