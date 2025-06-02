import streamlit as st
from topic_modeler import TopicModeler  # o importa tus funciones desde donde estén
from priority_model import predict_priority  # suponiendo que tengas un modelo de prioridad aparte

# Cargar modelos
topic_modeler = TopicModeler()
topic_modeler.load_model("path/a/tu/modelo")  # si aplica

st.title("Clasificador de Tickets de TI")

# Entrada de texto
ticket = st.text_area("Escribe el texto del ticket (body):", height=200)

if st.button("Clasificar"):
    if ticket.strip() == "":
        st.warning("Por favor ingresa un texto.")
    else:
        # Procesar texto
        clean_text = topic_modeler.preprocess_text(ticket)
        topic = topic_modeler.predict_topic(clean_text)
        subtopic = topic_modeler.predict_subtopic(clean_text)
        priority = predict_priority(clean_text)

        st.subheader("Resultados:")
        st.markdown(f"**Tópico:** {topic}")
        st.markdown(f"**Subtópico:** {subtopic}")
        st.markdown(f"**Prioridad:** {priority}")
