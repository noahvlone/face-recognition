import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# Load model
model = tf.keras.models.load_model("model_cnn_klasifikasi.h5")

# Label kelas
class_labels = ['agung', 'farhan']  # ganti sesuai urutan train_generator.class_indices

# --- App UI Modern ---
st.set_page_config(
    page_title="Agung vs Farhan Detector",
    page_icon=":camera_with_flash:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .header-style {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .tab-container {
        display: flex;
        justify-content: center;
        margin-bottom: 25px;
    }
    
    .tab {
        padding: 12px 24px;
        background: #f0f5ff;
        border: none;
        border-radius: 30px;
        margin: 0 10px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .tab.active {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(37, 117, 252, 0.3);
    }
    
    .upload-area {
        border: 2px dashed #2575fc;
        border-radius: 20px;
        padding: 40px 20px;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 30px;
    }
    
    .camera-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        margin-bottom: 30px;
    }
    
    .capture-btn {
        display: block;
        width: 100%;
        padding: 14px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        margin-top: 20px;
        transition: all 0.3s ease;
    }
    
    .capture-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(37, 117, 252, 0.4);
    }
    
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edff 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(37, 117, 252, 0.1);
        margin-top: 30px;
    }
    
    .confidence-bar {
        height: 12px;
        background: #e0e7ff;
        border-radius: 10px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    
    footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        color: #6c757d;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Header dengan gradient modern
st.markdown('<p class="header-style">Agung vs Farhan Face Detector</p>', unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 18px; color: #495057;'>
    Upload a face photo or capture from webcam to detect <strong style='color:#6a11cb'>Agung</strong> or <strong style='color:#2575fc'>Farhan</strong>
</p>
""", unsafe_allow_html=True)

# Tab untuk pilihan input
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Capture from Webcam"])

# Inisialisasi state
if 'camera_capture' not in st.session_state:
    st.session_state.camera_capture = None
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# Tab 1: Upload Gambar
with tab1:
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "DRAG & DROP IMAGE HERE", 
        type=["jpg", "jpeg", "png"],
        key="file_uploader",
        label_visibility="collapsed"
    )
    st.markdown('<p style="color: #6c757d; margin-top: 10px;">Supported formats: JPG, JPEG, PNG</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        st.session_state.camera_capture = None

# Tab 2: Ambil dari Webcam
with tab2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <p style="color: #495057;">Position your face in the frame and click capture</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tombol untuk membuka kamera
    if not st.session_state.show_camera:
        if st.button('🎥 Open Camera', use_container_width=True):
            st.session_state.show_camera = True
    
    # Tampilkan komponen kamera
    if st.session_state.show_camera:
        camera_capture = st.camera_input("Take a picture", key="camera")
        
        if camera_capture:
            st.session_state.camera_capture = camera_capture
            st.session_state.show_camera = False
            st.rerun()  # PERBAIKAN DI SINI: ganti experimental_rerun dengan rerun
    
    # Tombol reset jika sudah ada capture
    if st.session_state.camera_capture:
        if st.button('🔄 Retake Photo', use_container_width=True):
            st.session_state.camera_capture = None
            st.session_state.show_camera = True
            st.rerun()  # PERBAIKAN DI SINI: ganti experimental_rerun dengan rerun

# Kolom untuk layout responsif
col1, col2 = st.columns([1, 1])

# Proses gambar yang dipilih (baik dari upload atau kamera)
input_image = uploaded_file if uploaded_file else st.session_state.camera_capture

if input_image:
    # Tampilkan gambar
    with col1:
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
        else:
            # PERBAIKAN TAMBAHAN: handle camera capture
            img = Image.open(st.session_state.camera_capture).convert("RGB")
        
        st.image(
            img, 
            caption="Your Image",
            use_container_width=True,
            output_format="JPEG",
        )
        
    with col2:
        # Animasi loading
        with st.spinner('Analyzing image...'):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = class_labels[predicted_index]
            confidence = prediction[predicted_index]
            
        # Tampilkan hasil dengan efek visual modern
        st.balloons()
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        
        # Icon berdasarkan hasil prediksi
        if predicted_label == 'agung':
            icon = "👑"
            color = "#6a11cb"
        else:
            icon = "🚀"
            color = "#2575fc"
            
        st.markdown(f"""
        <h2 style='text-align: center; color: {color};'>
            {icon} It's {predicted_label.capitalize()}!
        </h2>
        <p style='text-align: center; font-size: 20px; margin-top: -10px;'>
            Confidence: <strong>{confidence:.1%}</strong>
        </p>
        """, unsafe_allow_html=True)
        
        # Confidence bar animasi
        st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-fill" style="width: {confidence*100}%"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interpretasi confidence
        if confidence > 0.85:
            msg = "High confidence prediction"
        elif confidence > 0.7:
            msg = "Good confidence prediction"
        else:
            msg = "Moderate confidence prediction"
            
        st.markdown(f"<p style='text-align: center; color: #495057;'>{msg}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 14px;">
    <p>Powered by TensorFlow & Streamlit | AI Face Recognition System</p>
</div>
""", unsafe_allow_html=True)