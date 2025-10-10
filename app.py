# ============ IMPORTS ============
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json
import pandas as pd
from datetime import datetime, date   # thêm date
import os
import io
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai

# NEW: hiển thị 2 cột rộng rãi
st.set_page_config(page_title="Phân loại vết thương", layout="wide")

# ============ HÀM NHÚNG YOUTUBE ============
def render_youtube(url, height=460):
    """Nhúng video YouTube bằng iframe, kiểm soát kích thước."""
    if "youtube.com" in url or "youtu.be" in url:
        if "youtu.be" in url:
            vid_id = url.split("/")[-1]
        else:
            vid_id = url.split("v=")[-1].split("&")[0]
        st.markdown(f'''
        <iframe width="100%" height="{height}" 
                src="https://www.youtube.com/embed/{vid_id}?rel=0" 
                frameborder="0" allowfullscreen></iframe>
        ''', unsafe_allow_html=True)
    else:
        st.video(url)

# ============ LOAD FILE THÔNG TIN ============
with open("wound_info.json", "r", encoding="utf-8") as f:
    wound_info = json.load(f)

# ============ LOAD MODEL ============
MODEL_PATH = "model_class.keras"
model = load_model(MODEL_PATH)

# ============ DANH SÁCH CLASS ============
class_names = [
    'Abrasions', 'Bruises', 'Burns', 'Cut',
    'Diabetic Wounds', 'Laceration', 'Normal',
    'Pressure Wounds', 'Surgical Wounds', 'Venous Wounds'
]

# ============ HÀM DỰ ĐOÁN ============
def predict_image(image: Image.Image):
    img_rgb = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_rgb, (160, 160))
    input_img = np.expand_dims(img_resized, axis=0)

    predictions = model.predict(input_img)
    scores = tf.nn.softmax(predictions[0]).numpy()
    pred_label = np.argmax(scores)
    confidence = scores[pred_label]

    return class_names[pred_label], confidence, scores

# ================== FIRST AID VIDEOS ==================
first_aid_videos = {
    "Abrasions": [
        "https://www.youtube.com/watch?v=K9Loqqoe3pE",  # Trầy xước - VTC14
        "https://www.youtube.com/watch?v=IKcddYDbm6s",  # Kỹ năng xử lý vết trầy xước
        "https://www.youtube.com/watch?v=zIo41ndcKng"   # Tránh sẹo lồi sau trầy xước
    ],

    "Cut": [
        "https://www.youtube.com/watch?v=T-8Ac6RYZgI",  # Sơ cứu vết cắt sâu
        "https://www.youtube.com/watch?v=1c6csZihf8M"   # Cách cầm máu vết thương
    ],

    "Burns": [
        "https://www.youtube.com/watch?v=EWH6LA5Cg2Y",  # Sơ cứu vết bỏng
        "https://www.youtube.com/watch?v=J0ZtLnhpJxM"   # Xử lý vết bỏng an toàn, tránh sẹo
    ],

    "Normal": [
        "https://www.youtube.com/watch?v=98yNdXf_Yw4"   # Chăm sóc da và xử trí vết thương thường
    ],

    "Laceration": [
        "https://www.youtube.com/watch?v=R4DutP4ya9w",  # Cắt chỉ vết thương
        "https://www.youtube.com/watch?v=tRpHkcjgkdw"   # Mẹo cắt chỉ an toàn
    ],

    "Diabetic Wounds": [
        "https://www.youtube.com/watch?v=iPgnjJICzos",  # Vì sao vết thương tiểu đường dễ nhiễm trùng
        "https://www.youtube.com/watch?v=IHlV2M-8a2Q",  # Chăm sóc vết thương người tiểu đường
        "https://www.youtube.com/watch?v=N8M4qj6w82c"   # Vệ sinh vết thương đái tháo đường
    ],

    "Pressure Wounds": [
        "https://www.youtube.com/watch?v=BiqLzQ6nHWM",  # Hướng dẫn phòng & chăm sóc loét tỳ đè
        "https://www.youtube.com/watch?v=ynP1Ocooh3E"   # Cách thay băng vết loét tỳ đè
    ],

    "Surgical Wounds": [
        "https://www.youtube.com/watch?v=R4DutP4ya9w",  # Cắt chỉ sau phẫu thuật
        "https://www.youtube.com/watch?v=tRpHkcjgkdw"   # Hướng dẫn chăm sóc vết mổ
    ],

    "Bruises": [
        "https://www.youtube.com/watch?v=4E_tScuwOYo",  # Cách xử lý vết bầm tím
        "https://www.youtube.com/watch?v=0uZpPpxM1aE"   # Mẹo giảm sưng và tan máu bầm
    ],

    "Venous Wounds": [
        "https://www.youtube.com/watch?v=4ZUXy1kJqDg",  # Vết loét tĩnh mạch và cách chăm sóc
        "https://www.youtube.com/watch?v=nL-H3RSlFOQ"   # Hướng dẫn điều trị loét chi dưới
    ],
}


def get_first_aid_videos(label: str):
    return first_aid_videos.get(label, [])

# ============ UI MỞ ĐẦU ============
st.title("🩺 Chương trình phân loại vết thương")
st.write("Upload ảnh chụp để chương trình dự đoán loại vết thương & gợi ý phương án sơ cứu.")
st.divider()

if "wound_records" not in st.session_state:
    st.session_state.wound_records = []

# ========== Khu vực upload + dự đoán ==========
uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã upload", use_container_width=True)

    # Dự đoán
    label, confidence, scores = predict_image(image)
    info = wound_info[label]

    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"### ✅ Kết quả: **{info['ten_viet']} ({label})**")
        st.markdown(f"**Độ tin cậy:** {confidence*100:.2f}%")

        with st.expander("📋 Mô tả chi tiết", expanded=True):
            st.write(info["mo_ta"])

        with st.expander("🚑 Sơ cứu / Điều trị ban đầu", expanded=True):
            st.write(info["so_cuu"])

    with right:
        st.subheader("🎬 Video sơ cứu")
        vids = get_first_aid_videos(label)
        if not vids:
            st.info("Chưa có video cho loại vết thương này.")
        else:
            if len(vids) > 1:
                idx = st.selectbox("Chọn video:", range(len(vids)), format_func=lambda x: f"Video {x+1}")
            else:
                idx = 0
            render_youtube(vids[idx], height=460)

    st.divider()

    # ========== 📝 KHU VỰC NHẬT KÝ ==========
    st.subheader("📝 Nhật ký bệnh án")

    with st.form("wound_log_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            pid = st.text_input("Mã BN / Họ tên", placeholder="VD: BN_2025_001 hoặc Nguyễn Văn A")
            age = st.text_input("Tuổi", placeholder="VD: 32")
            phone = st.text_input("SĐT (tuỳ chọn)", placeholder="VD: 09xx...")
        with c2:
            severity = st.selectbox("Mức độ", ["Nhẹ", "Vừa", "Nặng"], index=1)
            final_label = st.text_input("Chẩn đoán (có thể chỉnh)", value=f"{info['ten_viet']} ({label})")
            followup = st.date_input("Ngày tái khám (tuỳ chọn)", value=date.today())

        symptoms = st.text_area("Triệu chứng/diễn tiến", placeholder="Mô tả chi tiết...")
        treatment = st.text_area("Xử trí/điều trị đã thực hiện", value=info["so_cuu"])
        notes = st.text_area("Ghi chú khác", placeholder="Dị ứng thuốc, bệnh nền...")

        submitted = st.form_submit_button("💾 Lưu nhật ký ca này")
        if submitted:
            record = {
                "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Mã/BN": pid,
                "Tuổi": age,
                "SĐT": phone,
                "Mức độ": severity,
                "Chẩn đoán (cuối)": final_label,
                "Chẩn đoán (AI)": f"{info['ten_viet']} ({label})",
                "Độ tin cậy (%)": round(confidence*100, 2),
                "Triệu chứng": symptoms,
                "Xử trí ban đầu": treatment,
                "Ghi chú": notes,
                "Ngày tái khám": followup.strftime("%Y-%m-%d") if followup else "",
                "Tên file ảnh": getattr(uploaded_file, "name", ""),
            }
            st.session_state.wound_records.append(record)
            st.success("Đã lưu vào nhật ký! (ở bên dưới)")

    if st.session_state.wound_records:
        st.markdown("#### 📚 Lịch sử nhật ký")
        df = pd.DataFrame(st.session_state.wound_records)
        st.dataframe(df, use_container_width=True, height=340)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Tải toàn bộ nhật ký (CSV)", data=csv_bytes, file_name="wound_log.csv", mime="text/csv")
else:
    st.info("Hãy chọn một ảnh để bắt đầu chẩn đoán và ghi nhật ký.")

import requests

st.divider()
st.header("💬 Tư vấn AI - Bác sĩ ảo   ")

# ===== Khởi tạo bộ nhớ hội thoại =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===== Hiển thị lịch sử hội thoại =====
for chat in st.session_state.chat_history:
    role = "🧑‍⚕️ Bác sĩ AI" if chat["role"] == "assistant" else "🧍‍♂️ Bạn"
    with st.chat_message(chat["role"]):
        st.markdown(f"**{role}:** {chat['content']}")

# ===== Nhập câu hỏi =====
user_input = st.chat_input("Nhập câu hỏi của bạn (vd: Vết thương này có cần băng lại không?)")

if user_input:
    # Hiển thị câu hỏi của người dùng
    st.chat_message("user").markdown(f"**Bạn:** {user_input}")
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ===== GỌI OLLAMA GEMMA =====
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma:2b",    # 👈 Model bạn đang dùng
                "prompt": user_input,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        ai_text = data.get("response", "").strip()

        # Hiển thị phản hồi từ AI
        st.chat_message("assistant").markdown(f"**🧑‍⚕️ Bác sĩ AI:** {ai_text}")
        st.session_state.chat_history.append({"role": "assistant", "content": ai_text})

    except requests.exceptions.RequestException as e:
        st.error(f"🚫 Lỗi khi kết nối Ollama: {e}")
        st.info("⚙️ Kiểm tra lại: Đã chạy `ollama serve` và tải model `gemma3:2b` bằng `ollama pull gemma3:2b` chưa?")


