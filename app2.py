import streamlit as st

# Configurar página
st.set_page_config(page_title="Cambio de Fondo", layout="wide")

# Inicializar estado de fondo si no existe
if 'bg_color' not in st.session_state:
    st.session_state.bg_color = 'white'

# Botón para cambiar fondo a azul
if st.button('Cambiar fondo a azul'):
    st.session_state.bg_color = 'blue'

# Inyectar CSS para cambiar el color de fondo
st.markdown(
    f"""
    <style>
    body {{
        background-color: {st.session_state.bg_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Contenido de la app
st.title("Demo: Cambiar Fondo con Botón")
st.write("Presiona el botón de arriba para cambiar el fondo a azul.")
