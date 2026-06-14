import streamlit as st

# 1. Konfigurasi Halaman (Harus di paling atas)
st.set_page_config(
    page_title="PneumoScan AI Dashboard",
    page_icon="🫁",
    layout="wide"
)

import torch
import numpy as np
import cv2
import pandas as pd
import pydicom
import time
import os
import gdown
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou
from model_architecture_02 import create_efficientnet_b0_fasterrcnn

# =======================
# LOAD MODEL & CSV
# =======================
@st.cache_resource
def load_model():
    model_path = "checkpoint_epoch_2.pth"
    
    # Memeriksa apakah file model sudah ada di server Cloud, jika belum akan diunduh otomatis
    if not os.path.exists(model_path):
        with st.spinner("Sedang mengunduh bobot model dari Google Drive (ini hanya dilakukan sekali)..."):
            file_id = "1rGJUG342YuZQb8rz17KfkRN52_rfJ2ZP"
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                st.stop()
                
    model = create_efficientnet_b0_fasterrcnn(num_classes=2)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

@st.cache_data
def load_gt_csv():
    try:
        df = pd.read_csv("stage_2_train_labels.csv", sep=';', encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        return df
    except:
        return None

model = load_model()
df_gt = load_gt_csv()
device = torch.device("cpu")
model.to(device)

# =======================
# UTILS
# =======================
def apply_clahe_gray(img):
    clahe = cv2.createCLAHE(2.0, (8,8))
    return clahe.apply(img)

def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.5):
    if boxes.numel() == 0:
        return boxes, scores

    keep_boxes, keep_scores = [], []
    boxes, scores = boxes.clone(), scores.clone()

    while boxes.size(0) > 0:
        max_idx = torch.argmax(scores)
        max_box = boxes[max_idx]
        max_score = scores[max_idx]

        keep_boxes.append(max_box)
        keep_scores.append(max_score)

        boxes = torch.cat([boxes[:max_idx], boxes[max_idx+1:]])
        scores = torch.cat([scores[:max_idx], scores[max_idx+1:]])

        if boxes.size(0) == 0:
            break

        ious = box_iou(max_box.unsqueeze(0), boxes).squeeze(0)
        scores = scores * torch.exp(-(ious ** 2) / sigma)

        keep = scores > score_threshold
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.stack(keep_boxes), torch.stack(keep_scores)


def visualize_with_gt(image, gt_boxes=None, pred_boxes=None, scores=None, thr=0.5):
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(image, cmap='gray')

    # Ground Truth
    if gt_boxes is not None:
        for i, box in enumerate(gt_boxes):
            x,y,w,h = box
            ax.add_patch(patches.Rectangle((x,y), w,h,
                                           linewidth=3, edgecolor='#00FF00', facecolor='none'))
            ax.text(x, y-5, f"GT {i}", color='#00FF00', fontsize=12, weight='bold')
            
    # Prediksi
    if pred_boxes is not None:
        idx = 0
        for box, score in zip(pred_boxes, scores):
            if score >= thr:
                x1,y1,x2,y2 = box
                ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                               linewidth=3, edgecolor='#FF0000', facecolor='none'))
                ax.text(x1, y1-5, f"P {idx}", color='#FF0000', fontsize=12, weight='bold')
                idx += 1

    plt.axis('off')
    return fig

# =======================
# UI CUSTOM STYLING
# =======================
st.markdown("""
<style>
.main { background-color: #f5f7fb; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a, #2563eb);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
.header-box {
    background: linear-gradient(90deg, #2563eb, #10b981);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


# =======================
# SIDEBAR / NAVIGASI
# =======================
st.sidebar.header("📌 Navigasi")

# Menu Pilihan Halaman Utama & Tambahan Revisi
menu_pilihan = st.sidebar.radio(
    "Pilih Halaman:",
    ["🔍 Deteksi Pneumonia", "📖 Tata Cara Penggunaan", "ℹ️ Tentang Sistem"]
)

show_gt = st.sidebar.checkbox("Tampilkan Ground Truth", True)

if df_gt is not None:
    st.sidebar.success("✅ CSV Berhasil Dimuat")
else:
    st.sidebar.error("❌ CSV Tidak Ditemukan")

st.sidebar.markdown("---")
st.sidebar.info("🔬 Faster R-CNN + EfficientNet-B0")


# ==============================================================================
# ROUTING HALAMAN
# ==============================================================================

# --- HALAMAN 1: DETEKSI PNEUMONIA (TAMPILAN UTAMA ASLI KAMU) ---
if menu_pilihan == "🔍 Deteksi Pneumonia":
    
    # Banner Header Tetap Tampil di Halaman Utama Deteksi
    st.markdown("""
    <div class="header-box">
    <h1>🫁 PneumoScan AI</h1>
    <h4>Sistem Deteksi Pneumonia Berbasis Deep Learning</h4>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📤 Upload Citra X-ray")
    uploaded_file = st.file_uploader("Upload", type=["jpg","png","jpeg","bmp","dcm"])

    if uploaded_file:
        start_total = time.perf_counter()

        # LOAD IMAGE
        if uploaded_file.name.endswith(".dcm"):
            ds = pydicom.dcmread(uploaded_file)
            img = ds.pixel_array.astype(np.float32)
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        img_tensor = torch.tensor(img).float().unsqueeze(0) / 255.0
        img_display = cv2.resize(img, (1024,1024))
        img_display = apply_clahe_gray(img_display)

        # INFERENCE
        start_inference = time.perf_counter()

        with torch.no_grad():
            outputs = model([img_tensor.to(device)])

        p_boxes, p_scores = soft_nms(outputs[0]['boxes'], outputs[0]['scores'])
        p_boxes = p_boxes.cpu().numpy()
        p_scores = p_scores.cpu().numpy()

        end_inference = time.perf_counter()
        inference_time = end_inference - start_inference

        # SCALE BOX
        h,w = img.shape
        p_boxes[:, [0,2]] *= (1024/w)
        p_boxes[:, [1,3]] *= (1024/h)

        # GT
        gt_boxes = None
        if show_gt and df_gt is not None:
            pid = uploaded_file.name.split('.')[0]
            rows = df_gt[df_gt['patientId'] == pid]
            if not rows.empty:
                gt_boxes = rows[['x','y','width','height']].values

        # DETECTION
        threshold = 0.5
        detected = False
        bbox_pred_list = []

        for i in range(len(p_scores)):
            if p_scores[i] >= threshold:
                detected = True
                x1,y1,x2,y2 = p_boxes[i]
                bbox_pred_list.append({
                    "patientId": uploaded_file.name,
                    "x": round(float(x1),2),
                    "y": round(float(y1),2),
                    "width": round(float(x2-x1),2),
                    "height": round(float(y2-y1),2)
                })

        # VISUAL
        col1, col2 = st.columns([2,1])

        with col1:
            fig = visualize_with_gt(img_display, gt_boxes, p_boxes, p_scores, threshold)
            st.pyplot(fig)
            st.caption("🟢 Ground Truth | 🔴 Prediksi AI")

        with col2:
            if detected:
                st.error("⚠️ Area Indikasi Pneumonia Ditemukan")
            else:
                st.success("Tidak ditemukan area indikasi pneumonia")

        end_total = time.perf_counter()
        total_time = end_total - start_total

        print(f"DEBUG [{uploaded_file.name}] Inference: {inference_time:.3f}s | Total: {total_time:.3f}s")

        # TABEL 
        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Tabel Prediksi AI")
            if bbox_pred_list:
                st.dataframe(pd.DataFrame(bbox_pred_list), use_container_width=True)
            else:
                st.info("Tidak ada prediksi")

        with c2:
            st.subheader("🟢 Ground Truth")
            if gt_boxes is not None:
                st.dataframe(pd.DataFrame(gt_boxes, columns=['x','y','width','height']), use_container_width=True)
            else:
                st.info("Tidak tersedia")

    else:
        st.warning("Silakan upload gambar terlebih dahulu.")


# --- HALAMAN 2: REVISI TATA CARA PENGGUNAAN ---
elif menu_pilihan == "📖 Tata Cara Penggunaan":
    st.title("📖 Tata Cara Penggunaan PneumoScan AI")
    st.write("Ikuti panduan di bawah ini untuk mengoperasikan sistem deteksi:")
    
    with st.expander("Langkah 1: Persiapan Berkas Citra X-ray", expanded=True):
        st.write("Pastikan data rekam radiologi dada (Chest X-ray) pasien tersedia dalam format standar **JPG, JPEG, PNG, BMP**, atau format medis **DCM (DICOM)**.")
        
    with st.expander("Langkah 2: Proses Unggah Dokumen", expanded=True):
        st.write("Akses menu navigasi **🔍 Deteksi Pneumonia**, lalu gunakan area seret-taruh atau klik **'Browse files'** pada bagian unggah untuk memuat citra medis.")
        
    with st.expander("Langkah 3: Pembacaan Hasil Analisis", expanded=True):
        st.write("Tunggu beberapa saat hingga model AI selesai memproses gambar. Indikasi pneumonia akan ditandai dengan kotak pembatas (**🔴 Bounding Box**) beserta visualisasi perbandingan dengan data **🟢 Ground Truth** (jika diaktifkan).")


# --- HALAMAN 3: REVISI TENTANG SISTEM (ABOUT) ---
elif menu_pilihan == "ℹ️ Tentang Sistem":
    st.title("ℹ️ Tentang Sistem PneumoScan AI")
    
    st.markdown("""
    ### 🫁 Latar Belakang Proyek
    Sistem **PneumoScan AI** diimplementasikan guna mendeteksi keberadaan objek opasitas paru yang mengindikasikan penyakit Pneumonia melalui pencitraan X-ray dada.
    
    ### 🛠️ Spesifikasi Arsitektur & Dataset
    * **Algoritma Utama:** Faster R-CNN (Region-based Convolutional Neural Network)
    * **Ekstraktor Fitur / Backbone:** EfficientNet-B0
    * **Metode Kontras Gambar:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
    * **Dataset Referensi:** RSNA Pneumonia Detection Challenge
    """)
    
    st.divider()
    st.subheader("👨‍💻 Identitas Pengembang")
    
    col_img, col_txt = st.columns([1, 5])
    with col_img:
        st.image("foto_aku.png", width=110)
    with col_txt:
        st.markdown("""
        **TRI NOVITA** Jurusan Teknik Informatika  
        Universitas Lampung  
        
        *Sistem aplikasi dashboard ini dibangun menggunakan kerangka kerja Streamlit dan backend Deep Learning berbasis PyTorch sebagai bagian dari visualisasi hasil tugas akhir/penelitian.*
        """)

# Footer universal (Selalu muncul di bagian paling bawah halaman manapun)
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>© 2026 PneumoScan AI</div>", unsafe_allow_html=True)
