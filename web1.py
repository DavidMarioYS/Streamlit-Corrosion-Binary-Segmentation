import streamlit as st
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import os
import pandas as pd
import requests

# Pengaturan halaman
st.set_page_config(page_title="Responsive Dashboard", page_icon=":guardsman:", layout="wide")

# Fungsi fungsi
# URL file model di Google Drive
MobileUNet_Asset = "https://drive.google.com/file/id=1uBJF7UyTaGiilIDF3t2z9IU6sMzWfmuR"
MobileUNet_Corrosion = "https://drive.google.com/file/id=1NFUHzL9PGyGAeN-kVLP_h0jUsKo-Qw06"
FCN8_Asset = "https://drive.google.com/file/id=1Alr6TDBNQZ4JNjVCM69qhh-FjZ986w1F"
FCN8_Corrosion = "https://drive.google.com/file/id=1XnvchbaaYAiJp-VysBLEjr7jQx8ptwAP"
BiSeNetV2_Asset = "https://drive.google.com/file/id=1b2yvRkf3wwXQq1X25L4utfkPDT_0u6dC"
BiSeNetV2_Corrosion = "https://drive.google.com/file/id=1xuEdMDyY3Xz437FUEFclb90qy2MFvJyx"

@st.cache_data  # Cache untuk menghindari unduhan berulang
def download_model(url):
    # Unduh model dari Google Drive
    response = requests.get(url, stream=True)
    model_path = f"{model_name}.h5"
    with open(model_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    return model
    

def display_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        # Membaca gambar
        st.text("Processing image...")
        progress = st.progress(0)
        
        # Simulasi pemrosesan
        time.sleep(2)  # Misalnya proses selama 2 detik
        progress.progress(100)
        
        image = load_image(uploaded_file)
        
        # Membuat container dengan dua kolom
        with st.container():
            col1, col2 = st.columns(2)

            # Kolom pertama untuk visualisasi gambar
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            # Kolom kedua untuk informasi gambar
            with col2:
                # Mengambil informasi gambar
                resolution = f"{image.width} x {image.height}"
                color_mode = image.mode
                image_format = image.format

                # Membuat card informasi
                st.markdown(
                    f"""
                    <div style="padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
                        <h4>Image Details</h4>
                        <ul>
                            <li><b>Resolution:</b> {resolution}</li>
                            <li><b>Color Mode:</b> {color_mode}</li>
                            <li><b>Image Format:</b> {image_format}</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.write("Silahkan Upload Gambar Terlebih Dahulu!")
    return

def load_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None
    
def preprocessing(uploaded_file):
    if uploaded_file is not None:
        # Menampilkan button untuk preprocessing
        if st.button("Preprocessing"):
            # Membaca file yang diunggah
            # Membaca gambar
            st.text("Processing image...")
            progress = st.progress(0)
            
            # Simulasi pemrosesan
            time.sleep(2)  # Misalnya proses selama 2 detik
            progress.progress(100)
            
            image = load_image(uploaded_file)

            # Resize gambar ke ukuran 128x128
            resized_image = image.resize((128, 128))
            
            # Konversi gambar ke array uint8
            image_array = np.array(resized_image, dtype=np.uint8)
            
            # Menyimpan array ke st.session_state
            st.session_state.image_array = image_array
            
            # Menampilkan informasi array gambar
            st.write("Dimensi Array Gambar:", image_array.shape)  # Menampilkan dimensi
            st.write("Array Gambar (Sample Channel):", image_array[:, :, 0])  # Hanya channel R
            
            # Validasi apakah gambar berhasil dimuat
            if image_array is None:
                st.error("Error loading image.")
                return
            else:
                # Tampilkan gambar hasil unggahan
                st.image(image, caption=f"Image Resize {image_array.shape}", use_container_width=True)
                st.success("Image uploaded and processed successfully!")
            
            st.write("Gambar berhasil dimuat dalam format RGB.")

    else:
        st.info("Please upload an image file to proceed.")

    return

def model():
    # Membuat 4 kolom untuk tampilan tab
    col1, col2, col3 = st.columns(3)
    
    # Tombol dan status tab
    if "tab1_selected" not in st.session_state:
        st.session_state.tab1_selected = False
    if "tab2_selected" not in st.session_state:
        st.session_state.tab2_selected = False
    if "tab3_selected" not in st.session_state:
        st.session_state.tab3_selected = False

    # Menambahkan styling untuk tombol menggunakan HTML dan CSS
    st.markdown("""
        <style>
            .stButton button {
                width: 100%;
                height: 60px;
                font-size: 18px;
                background-color.active:
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Menambahkan tombol atau teks di setiap kolom sebagai tab
    with col1:
        tab1 = st.button("🔴 Mobile U-Net")
        if tab1:
            st.session_state.tab1_selected = True
            st.session_state.tab2_selected = False
            st.session_state.tab3_selected = False

    with col2:
        tab2 = st.button("🟢 FCN8")
        if tab2:
            st.session_state.tab2_selected = True
            st.session_state.tab1_selected = False
            st.session_state.tab3_selected = False
            
    with col3:
        tab3 = st.button("🔵 BiSeNetV2")
        if tab3:
            st.session_state.tab3_selected = True
            st.session_state.tab1_selected = False
            st.session_state.tab2_selected = False
    
    # Periksa apakah image_array tersedia
    if "image_array" not in st.session_state:
        st.warning("Please upload and preprocess an image first.")
        return
    
    # Ambil image_array dari session_state
    image_array = st.session_state.image_array

    # Konten berdasarkan tab yang dipilih
    if st.session_state.tab1_selected:
        st.write("Mobile U-Net tab is selected.")
        assetmodel = download_model(MobileUNet_Asset)
        corrosionmodel = download_model(MobileUNet_Corrosion)
        asset_model = load_keras_model(assetmodel)
        corrosion_model = load_keras_model(corrosionmodel)
        
        st.text("Pilih Proses Segmentasi...")
        
        Prediksi_Asset, Prediksi_Corrosion, Prediksi_Asset_pada_Corrosion = st.columns(3)
        
        # Menambahkan tombol atau teks di setiap kolom sebagai tab
        with Prediksi_Asset:
            tab_asset = st.button("Prediksi Asset")

        with Prediksi_Corrosion:
            tab_corrosion = st.button("Prediksi Corrosion")

        with Prediksi_Asset_pada_Corrosion:
            tab_asset_pada_corrosion = st.button("Prediksi Asset pada Corrosion")
        
        if tab_asset:
            ### Asset
            prediction = predict_image(image_array, asset_model, asset_colormap)
            
            # Overlay hasil segmentasi
            overlay_image = overlay_segmentation(image_array, prediction)
            
            # Menampilkan hasil segmentasi
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_array, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay_image, caption="Segmented Image", use_container_width=True)

            # Menyediakan tombol untuk mengunduh gambar hasil segmentasi
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            btn = st.download_button(
                label="Download Segmented Image",
                data=BytesIO(img_encoded.tobytes()),
                file_name="segmented_image.jpg",
                mime="image/jpeg"
            )
            
        if tab_corrosion:
            ### Corrosion
            st.text("Memproses Segmentasi Korosi...")
            # Prediksi segmentasi
            prediction = predict_image(image_array, corrosion_model, corrosion_colormap)
            
            # Overlay hasil segmentasi
            overlay_image = overlay_segmentation(image_array, prediction)
            
            # Menampilkan hasil segmentasi
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_array, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay_image, caption="Segmented Image", use_container_width=True)

            # Menyediakan tombol untuk mengunduh gambar hasil segmentasi
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            btn = st.download_button(
                label="Download Segmented Image",
                data=BytesIO(img_encoded.tobytes()),
                file_name="segmented_image.jpg",
                mime="image/jpeg"
            )
            
        if tab_asset_pada_corrosion:
            ### Asset pada Korosi
            st.text("Memproses Segmentasi Asset pada Korosi...")
            
            asset_prediction = predict_image(image_array, asset_model, asset_colormap)
            corrosion_prediction = predict_image(image_array, corrosion_model, corrosion_colormap)

            # Merge segmentations and detect non-asset corrosion areas
            merged_segmentation, corrosion_of_asset, corrosion_non_asset_mask = merge_segmentations(asset_prediction, corrosion_prediction)
            
            # Create a segmentation for corrosion outside of asset (yellow areas)
            corrosion_out_of_asset = np.zeros_like(corrosion_prediction)
            corrosion_out_of_asset[corrosion_non_asset_mask] = corrosion_non_asset_color  # Yellow
            overlay_image = overlay_segmentation(image_array, merged_segmentation, alpha=0.5)

            display_results(image_array, asset_prediction, corrosion_prediction, merged_segmentation, corrosion_of_asset, corrosion_out_of_asset, overlay_image)
            

    elif st.session_state.tab2_selected:
        st.write("FCN8 tab is selected.")
        st.write("Mobile U-Net tab is selected.")
        assetmodel = download_model(FCN8_Asset)
        corrosionmodel = download_model(FCN8_Corrosion)
        asset_model = load_keras_model(assetmodel)
        corrosion_model = load_keras_model(corrosionmodel)
        
        st.text("Pilih Proses Segmentasi...")
        
        Prediksi_Asset, Prediksi_Corrosion, Prediksi_Asset_pada_Corrosion = st.columns(3)
        
        # Menambahkan tombol atau teks di setiap kolom sebagai tab
        with Prediksi_Asset:
            tab_asset = st.button("Prediksi Asset")

        with Prediksi_Corrosion:
            tab_corrosion = st.button("Prediksi Corrosion")

        with Prediksi_Asset_pada_Corrosion:
            tab_asset_pada_corrosion = st.button("Prediksi Asset pada Corrosion")
        
        if tab_asset:
            ### Asset
            prediction = predict_image(image_array, asset_model, asset_colormap)
            
            # Overlay hasil segmentasi
            overlay_image = overlay_segmentation(image_array, prediction)
            
            # Menampilkan hasil segmentasi
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_array, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay_image, caption="Segmented Image", use_container_width=True)

            # Menyediakan tombol untuk mengunduh gambar hasil segmentasi
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            btn = st.download_button(
                label="Download Segmented Image",
                data=BytesIO(img_encoded.tobytes()),
                file_name="segmented_image.jpg",
                mime="image/jpeg"
            )
            
        if tab_corrosion:
            ### Corrosion
            st.text("Memproses Segmentasi Korosi...")
            # Prediksi segmentasi
            prediction = predict_image(image_array, corrosion_model, corrosion_colormap)
            
            # Overlay hasil segmentasi
            overlay_image = overlay_segmentation(image_array, prediction)
            
            # Menampilkan hasil segmentasi
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_array, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay_image, caption="Segmented Image", use_container_width=True)

            # Menyediakan tombol untuk mengunduh gambar hasil segmentasi
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            btn = st.download_button(
                label="Download Segmented Image",
                data=BytesIO(img_encoded.tobytes()),
                file_name="segmented_image.jpg",
                mime="image/jpeg"
            )
            
        if tab_asset_pada_corrosion:
            ### Asset pada Korosi
            st.text("Memproses Segmentasi Asset pada Korosi...")
            
            asset_prediction = predict_image(image_array, asset_model, asset_colormap)
            corrosion_prediction = predict_image(image_array, corrosion_model, corrosion_colormap)

            # Merge segmentations and detect non-asset corrosion areas
            merged_segmentation, corrosion_of_asset, corrosion_non_asset_mask = merge_segmentations(asset_prediction, corrosion_prediction)
            
            # Create a segmentation for corrosion outside of asset (yellow areas)
            corrosion_out_of_asset = np.zeros_like(corrosion_prediction)
            corrosion_out_of_asset[corrosion_non_asset_mask] = corrosion_non_asset_color  # Yellow
            overlay_image = overlay_segmentation(image_array, merged_segmentation, alpha=0.5)

            display_results(image_array, asset_prediction, corrosion_prediction, merged_segmentation, corrosion_of_asset, corrosion_out_of_asset, overlay_image)
            

    elif st.session_state.tab3_selected:
        st.write("BiSeNetV2 tab is selected.")
        st.write("Mobile U-Net tab is selected.")
        assetmodel = download_model(BiSeNetV2_Asset)
        corrosionmodel = download_model(BiSeNetV2_Corrosion)
        asset_model = load_keras_model(assetmodel)
        corrosion_model = load_keras_model(corrosionmodel)
        
        st.text("Pilih Proses Segmentasi...")
        
        Prediksi_Asset, Prediksi_Corrosion, Prediksi_Asset_pada_Corrosion = st.columns(3)
        
        # Menambahkan tombol atau teks di setiap kolom sebagai tab
        with Prediksi_Asset:
            tab_asset = st.button("Prediksi Asset")

        with Prediksi_Corrosion:
            tab_corrosion = st.button("Prediksi Corrosion")

        with Prediksi_Asset_pada_Corrosion:
            tab_asset_pada_corrosion = st.button("Prediksi Asset pada Corrosion")
        
        if tab_asset:
            ### Asset
            prediction = predict_image(image_array, asset_model, asset_colormap)
            
            # Overlay hasil segmentasi
            overlay_image = overlay_segmentation(image_array, prediction)
            
            # Menampilkan hasil segmentasi
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_array, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay_image, caption="Segmented Image", use_container_width=True)

            # Menyediakan tombol untuk mengunduh gambar hasil segmentasi
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            btn = st.download_button(
                label="Download Segmented Image",
                data=BytesIO(img_encoded.tobytes()),
                file_name="segmented_image.jpg",
                mime="image/jpeg"
            )
            
        if tab_corrosion:
            ### Corrosion
            st.text("Memproses Segmentasi Korosi...")
            # Prediksi segmentasi
            prediction = predict_image(image_array, corrosion_model, corrosion_colormap)
            
            # Overlay hasil segmentasi
            overlay_image = overlay_segmentation(image_array, prediction)
            
            # Menampilkan hasil segmentasi
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_array, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay_image, caption="Segmented Image", use_container_width=True)

            # Menyediakan tombol untuk mengunduh gambar hasil segmentasi
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            btn = st.download_button(
                label="Download Segmented Image",
                data=BytesIO(img_encoded.tobytes()),
                file_name="segmented_image.jpg",
                mime="image/jpeg"
            )
            
        if tab_asset_pada_corrosion:
            ### Asset pada Korosi
            st.text("Memproses Segmentasi Asset pada Korosi...")
            
            asset_prediction = predict_image(image_array, asset_model, asset_colormap)
            corrosion_prediction = predict_image(image_array, corrosion_model, corrosion_colormap)

            # Merge segmentations and detect non-asset corrosion areas
            merged_segmentation, corrosion_of_asset, corrosion_non_asset_mask = merge_segmentations(asset_prediction, corrosion_prediction)
            
            # Create a segmentation for corrosion outside of asset (yellow areas)
            corrosion_out_of_asset = np.zeros_like(corrosion_prediction)
            corrosion_out_of_asset[corrosion_non_asset_mask] = corrosion_non_asset_color  # Yellow
            overlay_image = overlay_segmentation(image_array, merged_segmentation, alpha=0.5)

            display_results(image_array, asset_prediction, corrosion_prediction, merged_segmentation, corrosion_of_asset, corrosion_out_of_asset, overlay_image)
            
    
    return

# Fungsi untuk memuat model TensorFlow Keras
def load_keras_model(model_path):
    """Fungsi untuk memuat model Keras dari path."""
    if not os.path.exists(model_path):
        st.error("Model file tidak ditemukan.")
        return None
    model = load_model(model_path)
    return model

# Define colormap for visualization
asset_colormap = {
    0: [0, 0, 0],    # Black for background
    1: [0, 0, 255]   # Blue for asset
}

corrosion_colormap = {
    0: [0, 0, 0],    # Black for background
    1: [255, 0, 0]   # Red for corrosion
}

class_color = {
    0: [0, 0, 0], 
    1: [255, 255, 255]}  # Class color for segmentation

# Define colormap for merged segmentation (including overlap color)
overlap_color_white = [255, 255, 255]  # White for overlap (corrosion + asset)
corrosion_non_asset_color = [255, 255, 0]  # Yellow for corrosion + non-asset

# Preprocessing and utility functions
def predict_image(image, model, colormap):
    """Predict image using the given model and apply specific class colors."""
    input_image = cv2.resize(image, (128, 128))  # Resize image to match model input size
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    prediction = model.predict(input_image)  # Get model prediction
    predicted_class = np.argmax(prediction[0], axis=-1)  # Get the class index with max probability

    # Create an RGB prediction map
    rgb_prediction = np.zeros((128, 128, 3), dtype=np.uint8)
    for class_index, color in colormap.items():
        rgb_prediction[predicted_class == class_index] = color

    return rgb_prediction

def merge_segmentations(asset_segmentation, corrosion_segmentation):
    """Merge asset and corrosion segmentations into a single image, only showing corrosion within asset."""
    # Create masks for each interaction
    asset_mask = np.all(asset_segmentation == [0, 0, 255], axis=-1)  # Asset pixels (Blue)
    corrosion_mask = np.all(corrosion_segmentation == [255, 0, 0], axis=-1)  # Corrosion pixels (Red)
    # Corrosion + Asset -> White (255, 255, 255)
    overlap_mask = np.logical_and(corrosion_mask, asset_mask)  # Intersection of corrosion and asset
    
    # Corrosion + Non-Asset -> Yellow (255, 255, 0)
    corrosion_non_asset_mask = np.logical_and(corrosion_mask, ~asset_mask)  # Corrosion but not asset

    # Initialize merged segmentation
    merged_segmentation = np.zeros_like(asset_segmentation)

    # Assign colors
    merged_segmentation[corrosion_mask & asset_mask] = overlap_color_white  # Corrosion + Asset -> White
    merged_segmentation[corrosion_non_asset_mask] = corrosion_non_asset_color  # Corrosion + Non-Asset -> Yellow
    merged_segmentation[~corrosion_mask & asset_mask] = [0, 0, 255]  # Asset alone -> Blue
    merged_segmentation[~corrosion_mask & ~asset_mask] = [0, 0, 0]  # Background -> Green
    
    # Corrosion of Asset: Only show the overlap (White), rest is black
    corrosion_of_asset = np.zeros_like(merged_segmentation)
    corrosion_of_asset[overlap_mask] = overlap_color_white  # Only keep white (overlap)

    return merged_segmentation, corrosion_of_asset, corrosion_non_asset_mask

def overlay_segmentation(original_image, segmentation, alpha=0.5):
    """Overlay segmentation onto the original image."""
    # Resize segmentation to match the original image size
    segmentation_resized = cv2.resize(segmentation, (original_image.shape[1], original_image.shape[0]))
    
    # Perform overlay
    return cv2.addWeighted(original_image, 1 - alpha, segmentation_resized, alpha, 0)


# Fungsi untuk menampilkan overlay dan informasi piksel
def display_results(image, asset_prediction, corrosion_prediction, merged_segmentation, corrosion_of_asset, corrosion_out_of_asset, overlay_image):
    # Menampilkan hasil segmentasi di Streamlit
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(asset_prediction, caption="Segmentasi Asset", use_container_width=True)
    with col3:
        st.image(corrosion_prediction, caption="Segmentasi Korosi", use_container_width=True)
    with col4:
        st.image(merged_segmentation, caption="Segmentasi Gabungan", use_container_width=True)
    with col5:
        st.image(corrosion_of_asset, caption="Segmentasi Korosi pada Asset", use_container_width=True)
    with col6:
        st.image(corrosion_out_of_asset, caption="Segmentasi Korosi luar Asset", use_container_width=True)
    with col7:
        st.image(overlay_image, caption="Overlay Segmentasi Gabungan", use_container_width=True)

    # Hitung informasi terkait piksel
    total_pixels = image.shape[0] * image.shape[1]

    # Jumlah piksel biru (Asset)
    blue_pixels = np.sum(np.all(asset_prediction == [0, 0, 255], axis=-1))

    # Jumlah piksel merah (Corrosion)
    red_pixels = np.sum(np.all(corrosion_prediction == [255, 0, 0], axis=-1))

    # Jumlah piksel putih (Overlap Asset dan Corrosion)
    white_pixels = np.sum(np.all(merged_segmentation == [255, 255, 255], axis=-1))

    # Jumlah piksel kuning (Corrosion Out of Asset)
    yellow_pixels = np.sum(np.all(corrosion_out_of_asset == [255, 255, 0], axis=-1))

    # Persentase piksel putih terhadap piksel biru
    if blue_pixels > 0:
        white_percentage = (white_pixels / blue_pixels) * 100
    else:
        white_percentage = 0

    # Menampilkan persentase dalam format dengan dua angka desimal (misalnya 12.34%)
    formatted_white_percentage = f"{white_percentage:.2f}%"  # Format dengan 2 angka desimal

    # Menampilkan tabel informasi piksel
    data = {
        "Keterangan": ["Jumlah Total Piksel", "Jumlah Piksel Biru (Asset)", "Jumlah Piksel Merah (Korosi)",
                    "Jumlah Piksel Putih (Korosi pada Asset)", "Jumlah Piksel Kuning (Korosi Luar Asset)",
                    "Persentase Piksel Putih (Korosi pada Asset)"],
        "Nilai": [total_pixels, blue_pixels, red_pixels, white_pixels, yellow_pixels, formatted_white_percentage]
    }

    df = pd.DataFrame(data)
    st.table(df)
    
# Menambahkan CSS untuk styling
st.markdown("""
    <style>
        /* Styling untuk header */
        header {
            width: 100%;
            background-color: #f4bc30; /* Contoh warna header */
            padding: 20px;
            color: white;
            text-align: center;
        }
        .stAppHeader {
            background-color: #002D62; /* Warna latar belakang gelap */
        }

        /* Styling untuk sidebar */
        /* Mengatur warna latar sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #002D62, #0f06FF); /* Gradasi biru tua ke oranye */
            color: #ecf0f1 !important;           /* Warna teks terang */
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p {
            color: #ecf0f1 !important;           /* Warna teks terang */
        }
        
        .sidebar-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 70%;
        }
              
        /* Styling untuk footer */
        footer {
        background-color: #0f06FF; /* Latar belakang footer */
        color: white;
        position: fixed;
        font-family: Verdana, Geneva, Tahoma, sans-serif;
        padding: 10px;
        font-size: 0.9em;
        bottom: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        text-align: center;
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.2);
    }
        
        /* Media Queries */
        /* Untuk ponsel kecil */
        @media (min-width: 320px) {
            header {
                font-size: 18px;
            }
            .css-1d391kg {
                width: 100%;
                padding: 10px;
            }
            .main {
                font-size: 14px;
            }
            footer {
                font-size: 12px;
            }
        }

        /* Untuk tablet biasa */
        @media (min-width: 768px) {
            header {
                font-size: 22px;
            }
            .css-1d391kg {
                width: 80%;
                padding: 20px;
            }
            .main {
                font-size: 16px;
            }
            footer {
                font-size: 14px;
            }
        }

        /* Untuk laptop kecil */
        @media (min-width: 1280px) {
            header {
                font-size: 26px;
            }
            .css-1d391kg {
                width: 75%;
                padding: 25px;
            }
            .main {
                font-size: 18px;
            }
            footer {
                font-size: 16px;
            }
        }
    </style>
""", unsafe_allow_html=True)

def show_about():
    # Menambahkan gambar ke main content (jika perlu)
    opening_path = "C:/Users/David Mario Yohanes/Documents/SEM 7/BINARY SEGMENTATION/STREAMLIT/latar.jpg"
    st.image(opening_path, caption='Pipa Kilang - Pertambangan', use_container_width=True)

    # Main content section
    st.markdown("<div class='main'><h2>Informasi Menu Dashboard</h2></div>", unsafe_allow_html=True)
    # Membuat 3 kolom dengan ukuran sama (1/3 layar)
    col1, col2 = st.columns(2)

    # About dengan Expander
    with col1:
        st.markdown("""
            <div style="background-color: #f4bc30; 
            padding: 16px; 
            border-radius: 10px; 
            text-align: center; 
            color: white;
                <h3 style="text-align: center;">About</h3>
            </div>

        """, unsafe_allow_html=True)

        # Menambahkan Expander di dalam About
        with st.expander("Apa yang dimaksud dengan korosi?"):
            st.write("""
                ---
                Korosi adalah proses degradasi atau **kerusakan material**, terutama logam, akibat reaksi kimia atau elektrokimia dengan lingkungan sekitarnya, seperti oksigen, air, atau bahan kimia lainnya. 
                Proses ini sering menghasilkan perubahan pada permukaan material, seperti pembentukan **karat pada besi**. 
                Korosi dapat mengurangi kekuatan, fungsi, dan umur material.
            """)
        with st.expander("Apa yang dimaksud dengan Segmentasi Gambar?"):
            st.write("""
                ---
                **Segmentasi citra** adalah proses membagi suatu gambar digital ke dalam beberapa wilayah atau objek berdasarkan karakteristik tertentu, seperti warna, intensitas, tekstur, atau bentuk. 
                Tujuannya adalah untuk mempermudah analisis atau identifikasi dengan mengisolasi area penting dari gambar, seperti objek atau fitur yang diinginkan. 
                Segmentasi sering digunakan dalam bidang seperti pengolahan gambar medis, visi komputer, dan analisis data spasial.
            """)
        with st.expander("Standarisasi apa saja yang digunakan?"):
            st.write("""
                ---
                Terdapat beberapa standarisasi dalam memproses model segmentasi, yaitu:
                - **Standarisasi Image**: 
                    - Resize: `128`
                    - Normalisasi
                - **Standarisasi Augmentasi**: 
                    - Rotasi: `30, 45, 60, 90`, 
                    - Flip: `horizontal dan vertical`, 
                    - Hue: `10, -10`,
                    - Saturation: `0.5, 1.0, 1.5, 2.0` 
                - **Standarisasi Colormap**: 
                    - Korosi dan Asset: `putih [255,255,255] #FFFFFF`
                    - Non-Korosi dan Non-Asset: `hitam [0, 0, 0] #000000
                    - Visualiasi Gabungan (Asset): `biru 0, 0, 255 #0000FF`
                    - Visualiasi Gabungan (Korosi): `merah 255, 0, 0 #FF0000`
                    - Visualiasi Gabungan (Korosi pada Asset): `putih 255, 255, 255 #FFFFFF`
                    - Visualiasi Gabungan (Korosi luar Asset): `kuning 255, 255, 0 #FFFF00`
                - **Standarisasi Backbone Arsitektur**: `ResNet-50`.
            """)
            
    # Segmentation dengan Expander
    with col2:
        st.markdown("""
            <div style="background-color: #f4bc30; 
            padding: 16px; 
            border-radius: 10px; 
            text-align: center; 
            color: white;
                <h3 style="text-align: center;">Segmentation</h3>
            </div>

        """, unsafe_allow_html=True)
        with st.expander("Apa saja model yang dimiliki?"):
            st.write("""
                ---
                1. **Mobile U-Net + ResNet-50**
                    - **Efisiensi Komputasi**: Mobile U-Net dirancang untuk kecepatan dan efisiensi, sangat cocok untuk perangkat dengan keterbatasan sumber daya.
                    - **Transfer Learning**: Menggunakan ResNet-50 sebagai backbone memungkinkan transfer learning yang kuat dengan kemampuan untuk mengekstrak fitur tingkat tinggi dari gambar.
                    - **Ringan dan Cepat**: Memiliki ukuran model yang lebih kecil dibandingkan dengan U-Net standar, namun tetap mempertahankan performa yang baik pada banyak dataset.

                2. **FCN-8 + ResNet-50**
                    - **Hasil Segmentasi Akurat**: FCN-8 menggunakan pendekatan fully convolutional yang memberikan segmentasi yang lebih halus dan detail dengan pengolahan citra multiskala.
                    - **Skalabilitas**: Mampu menangani citra dengan resolusi tinggi tanpa kehilangan banyak informasi.
                    - **Backbone yang Kuat**: Penggunaan ResNet-50 sebagai backbone memungkinkan ekstraksi fitur yang mendalam dan kemampuan generalisasi yang lebih baik.

                3. **BiSeNetV2 + ResNet-50**
                    - **Pengolahan Real-Time**: BiSeNetV2 dirancang untuk kecepatan pengolahan real-time, menjadikannya pilihan tepat untuk aplikasi dengan kebutuhan kecepatan tinggi.
                    - **Akurasi Tinggi**: Dengan kemampuan segmentasi dua jalur (bilateral), model ini memaksimalkan akurasi dalam mengidentifikasi batas objek yang kompleks.
                    - **Efisiensi Memori**: Dikenal dengan penggunaan memori yang lebih efisien sambil tetap mempertahankan akurasi segmentasi yang tinggi, menjadikannya ideal untuk aplikasi edge computing.
                    """)
        with st.expander("Bagaimana cara menggunakan model segmentasi?"):
            st.write("""
                ---
                Berikut cara menggunakan model pada segmentasi:
                
                1. Buka menu **:frame_with_picture: Segmentation Image**
                2. Input gambar terlebih dahulu
                3. Klik button **Preprocessing** untuk melakukan persiapan gambar
                4. Pilih jenis model yang tersedia
                5. Pilih jenis segmentasi yang ingin dicoba
                6. Hasil segmentasi akan muncul berdasarkan proses pilihan model yang digunakan.
            """)

        # Menambahkan Expander di dalam Segmentation

def show_segmentation():
    st.subheader("Segmentasi Citra Korosi")
    st.title("Upload and Display Image with Details")

    # Membaca upload file
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    display_uploaded_image(uploaded_file)
    preprocessing(uploaded_file)
    model()
    


# Header section
st.markdown("<header><h1>AIDA Phase 3: AI Segmentasi Korosi Pipa</h1></header>", unsafe_allow_html=True)

# Sidebar section

## Menambahkan gambar logo ke sidebar dengan class CSS
base_path = os.getcwd()
logo_path = os.path.join(base_path, "logo.png")
st.sidebar.image(logo_path)

## Sidebar Judul
st.sidebar.header(":camera: Image Segmentation")

## Deskripsi Singkat di Sidebar
st.sidebar.write("Projek prediksi korosi pada pipa")
## Menu dengan pilihan radio buttons (Menambahkan ikon secara manual)
menu_option = st.sidebar.radio(
    "Pilih Opsi",
    (":house: About", ":frame_with_picture: Segmentation Image")
)

## Menampilkan konten berdasarkan pilihan menu
if menu_option == ":house: About":
    show_about()

elif menu_option == ":frame_with_picture: Segmentation Image":
    show_segmentation()
else:
    st.write("Pengaturan dapat dikonfigurasi di sini. ")

# Footer section
st.markdown("<footer><p>Footer Content</p></footer>", unsafe_allow_html=True)
