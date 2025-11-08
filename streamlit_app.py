import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN Y CARGA DE MODELO ---
MODEL_PATH = 'models/model.keras' 
IMG_SIZE = 224
TARGET_CONV_LAYERS = ['conv2d_3', 'conv2d_4', 'conv2d_5'] 
CLASS_NAMES = ['Female (Mujer)', 'Male (Hombre)']

# Carga el modelo y crea la versión estable para Grad-CAM
@st.cache_resource
def load_and_prepare_model():
    st.info("Cargando y preparando modelos (solo la primera vez)...")
    try:
        model_container = tf.keras.models.load_model(MODEL_PATH)
        if isinstance(model_container.layers[0], tf.keras.Model):
            model_sequential_layers = model_container.layers[0] 
        else:
            model_sequential_layers = model_container
            
        img_input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = img_input_tensor
        
        for layer in model_sequential_layers.layers:
            if 'dropout' in layer.name.lower(): 
                continue
            x = layer(x)

        grad_cam_model = Model(
            inputs=img_input_tensor,
            outputs=x
        )
        st.success("Modelos cargados exitosamente.")
        return model_container, grad_cam_model

    except Exception as e:
        st.error(f"Error al cargar el modelo. Verifique la ruta {MODEL_PATH}: {e}")
        return None, None

# --- FUNCIONES DE INTERPRETABILIDAD (XAI) ---

@tf.function
def make_gradcam_heatmap(model, img_array_normalized, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array_normalized, training=False) 
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[[2], [0]])
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val = 1e-9
    heatmap = heatmap / max_val
    return heatmap

def make_saliency_map(model, img_array_normalized):
    img_array_normalized = tf.cast(img_array_normalized, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_array_normalized)
        preds = model(img_array_normalized, training=False)
        top_class_channel = preds[:, 0]

    grads = tape.gradient(top_class_channel, img_array_normalized)
    saliency = tf.math.abs(grads[0])
    saliency = tf.reduce_max(saliency, axis=-1)
    
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-9)
    return saliency.numpy()

def superimpose_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + img.astype(np.float32) * (1 - alpha)
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def plot_prediction_chart(prediction):
    prob_male = prediction[0][0]
    prob_female = 1.0 - prob_male
    probabilities = [prob_female, prob_male]
    
    fig, ax = plt.subplots(figsize=(6, 2))
    bars = ax.bar(CLASS_NAMES, probabilities, color=['skyblue', 'lightcoral'])
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probabilidad')
    ax.set_title('Predicción de Probabilidad por Clase')
    
    for bar, prob in zip(bars, probabilities):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{prob:.2%}', ha='center', va='bottom')

    st.pyplot(fig)


# --- FUNCIÓN PRINCIPAL DE STREAMLIT ---

def main():
    st.set_page_config(page_title="CNNs-XAI: Clasificación de Género", layout="wide")
    st.title("👨‍🔬 Clasificador de Género y Mapas XAI")
    st.markdown("Sube una imagen de rostro para obtener la predicción y los mapas de interpretabilidad.")
    
    model_container, grad_cam_model = load_and_prepare_model()
    
    if model_container is None:
        st.stop()

    file_upload = st.file_uploader(
        "Sube una imagen de rostro (JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if file_upload is not None:
        image = Image.open(file_upload).convert('RGB')
        img_resized_pil = image.resize((IMG_SIZE, IMG_SIZE))
        img_resized_np = np.array(img_resized_pil)
        
        img_tensor_for_pred = tf.expand_dims(img_resized_np, axis=0)
        img_tensor_for_xai = tf.cast(img_tensor_for_pred, tf.float32) / 255.0

        with st.spinner("Realizando predicción..."):
            prediction = model_container.predict(img_tensor_for_pred)
            
        col_img, col_pred = st.columns([1, 1])
        
        with col_img:
            st.subheader("🖼️ Imagen Cargada")
            st.image(img_resized_pil, caption=file_upload.name, use_column_width=True)
            
        with col_pred:
            st.subheader("🧠 Resultado de la Predicción")
            prob_male = prediction[0][0]
            if prob_male > 0.5:
                label = "Male (Hombre)"
                conf = prob_male * 100
            else:
                label = "Female (Mujer)"
                conf = (1.0 - prob_male) * 100

            st.metric(label="Clase Predicha", value=label)
            st.metric(label="Confianza", value=f"{conf:.2f}%")
            
            st.markdown("---")
            st.markdown("**Gráfica de Probabilidad**")
            plot_prediction_chart(prediction)
        
        st.divider()

        # 4. Análisis de Interpretabilidad (XAI)
        st.header("🔍 Análisis de Interpretabilidad")
        
        # --- GRAD-CAM POR CAPAS ---
        st.subheader("🔥 Mapas Grad-CAM por Nivel de Abstracción")
        cols_gc = st.columns(len(TARGET_CONV_LAYERS))

        for i, layer_name in enumerate(TARGET_CONV_LAYERS):
            with cols_gc[i]:
                st.markdown(f"**Capa: `{layer_name}`**")
                
                with st.spinner(f"Calculando Grad-CAM para {layer_name}..."):
                    heatmap_tensor = make_gradcam_heatmap(grad_cam_model, img_tensor_for_xai, layer_name)
                    heatmap_gc = heatmap_tensor.numpy()
                    
                    overlay_gc = superimpose_heatmap(img_resized_np, heatmap_gc)
                    st.image(overlay_gc, caption=f"Atención en {layer_name}", use_column_width=True)

        st.divider()
        
        # --- SALIENCY MAP ---
        st.subheader("💡 Saliency Map (Sensibilidad al Píxel)")
        col_sm, _ = st.columns([1, 2])
        
        with col_sm:
            with st.spinner("Calculando Saliency Map..."):
                heatmap_sm = make_saliency_map(grad_cam_model, img_tensor_for_xai)
                overlay_sm = superimpose_heatmap(img_resized_np, heatmap_sm)
                st.image(overlay_sm, caption="Influencia de píxeles individuales", use_column_width=True)

if __name__ == '__main__':
    main()
