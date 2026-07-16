import streamlit as st
from ultralytics import YOLO
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.transforms import functional as F
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import numpy as np
import os
 
st.set_page_config(page_title="Deteksi Biji Kopi", layout="wide")
 
st.title("☕ Deteksi Kualitas Biji Kopi")
st.markdown("Upload **banyak gambar** untuk analisis kualitas biji kopi")
 
# PILIH MODEL
model_type = st.sidebar.selectbox(
    "Pilih Model",
    ["YOLO", "Faster R-CNN", "SSD300 VGG16"]
)

# MODE DETEKSI: Single-Object atau Multi-Objek
DETECTION_MODE = st.sidebar.radio(
    "Mode Deteksi",
    ["Single-Object", "Multi-Objek"],
    help="Single-Object: model dilatih 1 biji per gambar. "
         "Multi-Objek: model dilatih pada dataset mosaic (banyak biji per gambar)."
)

# PATH MODEL PER MODE
MODEL_PATHS = {
    "Single-Object": {
        "YOLO":         "YoloV8.pt",
        "Faster R-CNN": "fasterrcnn_resnet50_fpn.pth",
        "SSD300 VGG16": "model_ssd300_vgg16.pth",
    },
    "Multi-Objek": {
        "YOLO":         "YoloV8_multi.pt",
        "Faster R-CNN": "fasterrcnn_multi.pth",
        "SSD300 VGG16": "ssd300_multi.pth",
    },
}

# WARNA PER CLASS
CLASS_COLORS = {
    "defect":    (255, 0,   0  ),
    "longberry": (0,   0,   255),
    "peaberry":  (0,   200, 0  ),
    "premium":   (255, 165, 0  ),
}
 
# THRESHOLD PER MODEL
YOLO_THRESHOLD = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
RCNN_THRESHOLD = st.sidebar.slider("Faster R-CNN Confidence Threshold", 0.0, 1.0, 0.85, 0.05)
SSD_THRESHOLD = st.sidebar.slider("SSD Confidence Threshold", 0.0, 1.0, 0.75, 0.05)
 
# LOAD MODEL
@st.cache_resource
def load_model(model_type, mode):
    path = MODEL_PATHS[mode][model_type]
    if model_type == "YOLO":
        model = YOLO(path)
        # Setelah retrain, model.names sudah sesuai dengan data.yaml
        # {0: 'defect', 1: 'longberry', 2: 'peaberry', 3: 'premium'}
        return model, model.names
 
    elif model_type == "Faster R-CNN":

        names = {
            1: "defect",
            2: "longberry",
            3: "peaberry",
            4: "premium"
        }

        num_classes = len(names) + 1   # + background

        model = fasterrcnn_resnet50_fpn(num_classes=num_classes)

        model.load_state_dict(
            torch.load(path, map_location="cpu")
        )

        model.eval()

        return model, names


    else:  # SSD300 VGG16

        names = {
            1: "defect",
            2: "longberry",
            3: "peaberry",
            4: "premium"
        }

        num_classes = len(names) + 1   # + background

        model = ssd300_vgg16(num_classes=num_classes)

        model.load_state_dict(
            torch.load(path, map_location="cpu")
        )

        model.eval()

        return model, names
 
# Pilih path sesuai mode; bila model multi belum tersedia, fallback ke single
_chosen = MODEL_PATHS[DETECTION_MODE][model_type]
if not os.path.exists(_chosen):
    st.sidebar.warning(
        f"Model '{_chosen}' belum tersedia untuk mode {DETECTION_MODE}. "
        "Sementara memakai model Single-Object."
    )
    effective_mode = "Single-Object"
else:
    effective_mode = DETECTION_MODE

st.sidebar.caption(f"Mode: {effective_mode} — memuat: {MODEL_PATHS[effective_mode][model_type]}")
model, names = load_model(model_type, effective_mode)
 
def draw_boxes_rcnn(img, boxes, labels, scores, names, threshold=0.8):
    """Gambar bounding box di atas image PIL untuk Faster R-CNN dan SSD"""
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
 
    class_list = []
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
 
        class_name = names.get(int(label), str(int(label)))
        class_list.append(class_name)
 
        color = CLASS_COLORS.get(class_name, (255, 255, 0))
        x1, y1, x2, y2 = box.tolist()
 
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{class_name} {score:.2f}"
 
        # Background label
        text_bbox = draw.textbbox((x1, y1), label_text)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label_text, fill="white")
 
    return img_draw, class_list
 
uploaded_files = st.file_uploader(
    "Upload gambar",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)
 
if uploaded_files:
    st.success(f"{len(uploaded_files)} gambar berhasil diupload")
 
    cols = st.columns(3)
 
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 3]:
            img = Image.open(uploaded_file).convert("RGB")
 
            st.markdown(f"### 📷 Gambar {idx+1}")
 
            # Tampilkan gambar original
            col_orig, col_result = st.columns(2)
 
            with col_orig:
                st.caption("Original")
                st.image(img, use_container_width=True)
 
                with st.spinner("🔍 Deteksi..."):

                    if model_type == "YOLO":

                        results = model(img)

                        boxes = results[0].boxes

                        if boxes is not None and len(boxes) > 0:

                            boxes_xyxy = boxes.xyxy.cpu()
                            scores     = boxes.conf.cpu()

                            # FIX YOLO LABEL INDEX
                            labels = boxes.cls.cpu().int()

                            # FILTER CONFIDENCE
                            keep = scores >= YOLO_THRESHOLD

                            boxes_xyxy = boxes_xyxy[keep]
                            scores     = scores[keep]
                            labels     = labels[keep]
                            # FILTER CONFIDENCE
                            keep = scores >= YOLO_THRESHOLD

                            boxes_xyxy = boxes_xyxy[keep]
                            scores     = scores[keep]
                            labels     = labels[keep]

                            # NMS YOLO
                            keep_idx = nms(
                                boxes_xyxy,
                                scores,
                                iou_threshold=0.25
                            )

                            boxes_xyxy = boxes_xyxy[keep_idx]
                            scores     = scores[keep_idx]
                            labels     = labels[keep_idx]

                            # DRAW MANUAL
                            result_img = img.copy()
                            draw = ImageDraw.Draw(result_img)

                            class_list = []

                            for box, score, label in zip(
                                boxes_xyxy,
                                scores,
                                labels
                            ):

                                # LABEL 0-3
                                class_name = names[int(label)]

                                class_list.append(class_name)

                                color = CLASS_COLORS.get(
                                    class_name,
                                    (255,255,0)
                                )

                                x1, y1, x2, y2 = box.tolist()

                                draw.rectangle(
                                    [x1, y1, x2, y2],
                                    outline=color,
                                    width=3
                                )

                                label_text = f"{class_name} {score:.2f}"

                                text_bbox = draw.textbbox(
                                    (x1, y1),
                                    label_text
                                )

                                draw.rectangle(
                                    text_bbox,
                                    fill=color
                                )

                                draw.text(
                                    (x1, y1),
                                    label_text,
                                    fill="white"
                                )

                        else:
                            result_img = img
                            class_list = []

                        with col_result:
                            st.caption("Hasil Deteksi")
                            st.image(result_img, use_container_width=True)
                    else:
                        # Faster R-CNN dan SSD menggunakan pipeline yang sama
                        img_tensor = F.to_tensor(img)

                        with torch.no_grad():

                            prediction = model([img_tensor])

                            boxes_pred  = prediction[0]["boxes"]
                            labels_pred = prediction[0]["labels"]
                            scores_pred = prediction[0]["scores"]

                            # THRESHOLD BERDASARKAN MODEL
                            if model_type == "Faster R-CNN":
                                threshold = RCNN_THRESHOLD
                            else:
                                threshold = SSD_THRESHOLD

                            # FILTER CONFIDENCE
                            keep = scores_pred >= threshold

                            boxes_pred  = boxes_pred[keep]
                            labels_pred = labels_pred[keep]
                            scores_pred = scores_pred[keep]

                            # NMS KHUSUS SSD
                            if model_type == "SSD300 VGG16":
                                keep_idx = nms(
                                    boxes_pred,
                                    scores_pred,
                                    iou_threshold=0.15
                                )
                            else:
                                keep_idx = nms(
                                    boxes_pred,
                                    scores_pred,
                                    iou_threshold=0.3
                                )

                            boxes_pred  = boxes_pred[keep_idx]
                            labels_pred = labels_pred[keep_idx]
                            scores_pred = scores_pred[keep_idx]

                            # FILTER BOX TERLALU BESAR (KHUSUS SSD)
                            if model_type == "SSD300 VGG16":

                                filtered_boxes = []
                                filtered_labels = []
                                filtered_scores = []

                                img_width, img_height = img.size

                                for box, label, score in zip(
                                    boxes_pred,
                                    labels_pred,
                                    scores_pred
                                ):

                                    x1, y1, x2, y2 = box.tolist()

                                    box_width  = x2 - x1
                                    box_height = y2 - y1

                                    # Skip jika terlalu besar
                                    if box_width > img_width * 0.45:
                                        continue

                                    if box_height > img_height * 0.90:
                                        continue

                                    filtered_boxes.append(box)
                                    filtered_labels.append(label)
                                    filtered_scores.append(score)

                                if len(filtered_boxes) > 0:
                                    boxes_pred  = torch.stack(filtered_boxes)
                                    labels_pred = torch.stack(filtered_labels)
                                    scores_pred = torch.stack(filtered_scores)

                            # AMBIL TOP BOX TERBAIK UNTUK SSD
                            if model_type == "SSD300 VGG16":

                                top_k = min(10, len(scores_pred))

                                sorted_idx = torch.argsort(
                                    scores_pred,
                                    descending=True
                                )[:top_k]

                                boxes_pred  = boxes_pred[sorted_idx]
                                labels_pred = labels_pred[sorted_idx]
                                scores_pred = scores_pred[sorted_idx]

                            result_img, class_list = draw_boxes_rcnn(
                                img,
                                boxes_pred,
                                labels_pred,
                                scores_pred,
                                names,
                                threshold=threshold
                            )

                        with col_result:
                            st.caption("Hasil Deteksi")
                            st.image(result_img, use_container_width=True)

                    # HITUNG JUMLAH
                    st.markdown("**Hasil:**")

                    if class_list:

                        counter = Counter(class_list)

                        for key, val in counter.items():

                            color = CLASS_COLORS.get(key, (0, 0, 0))

                            hex_color = "#{:02x}{:02x}{:02x}".format(*color)

                            st.markdown(
                                f'<span style="color:{hex_color}">●</span> **{key}**: {val}',
                                unsafe_allow_html=True
                            )

                    else:
                        st.caption("⚠️ Tidak ada deteksi")