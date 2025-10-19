# ============ IMPORTS ============
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json
import pandas as pd
from datetime import datetime, date
import os, io, re, requests
import unicodedata
import difflib
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import urllib.parse as _u

# ============ PAGE CONFIG ============
st.set_page_config(page_title="AI Sơ cứu & Dị ứng", layout="wide")
import streamlit as st

st.sidebar.header("👤 Thông tin người dùng")

with st.sidebar.form("user_info_form"):
    name = st.text_input("Họ và tên")
    age = st.number_input("Tuổi", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
    city = st.text_input("Tỉnh/Thành phố hiện tại", placeholder="VD: TP. Hồ Chí Minh")
    district = st.text_input("Quận/Huyện", placeholder="VD: Quận 1")
    ward = st.text_input("Phường/Xã", placeholder="VD: Phường Bến Nghé")
    health_condition = st.text_area("Tình trạng sức khỏe (nếu có)", 
                                    placeholder="VD: Tiểu đường, tim mạch, dị ứng thuốc...")

    submitted = st.form_submit_button("✅ Lưu thông tin")

if submitted:
    st.session_state.user_info = {
        "Họ tên": name,
        "Tuổi": age,
        "Giới tính": gender,
        "Tỉnh/Thành phố": city,
        "Quận/Huyện": district,
        "Phường/Xã": ward,
        "Tình trạng sức khỏe": health_condition
    }
    st.success("Đã lưu thông tin! 🎉")

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

# ============ LOAD MODEL (cache) ============
MODEL_PATH = "model_class.keras"

@st.cache_resource
def load_cls_model(path=MODEL_PATH):
    return load_model(path)

model = load_cls_model(MODEL_PATH)

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

    preds = model.predict(input_img)
    scores = tf.nn.softmax(preds[0]).numpy()  # nếu model đã softmax sẵn thì vẫn ok
    pred_label = int(np.argmax(scores))
    confidence = float(scores[pred_label])
    return class_names[pred_label], confidence, scores

# ================== FIRST AID VIDEOS ==================
first_aid_videos = {
    "Abrasions": [
        "https://www.youtube.com/watch?v=K9Loqqoe3pE",
        "https://www.youtube.com/watch?v=IKcddYDbm6s",
        "https://www.youtube.com/watch?v=zIo41ndcKng"
    ],
    "Cut": [
        "https://www.youtube.com/watch?v=T-8Ac6RYZgI",
        "https://www.youtube.com/watch?v=1c6csZihf8M"
    ],
    "Burns": [
        "https://www.youtube.com/watch?v=EWH6LA5Cg2Y",
        "https://www.youtube.com/watch?v=J0ZtLnhpJxM"
    ],
    "Normal": [
        "https://www.youtube.com/watch?v=98yNdXf_Yw4"
    ],
    "Laceration": [
        "https://www.youtube.com/watch?v=R4DutP4ya9w",
        "https://www.youtube.com/watch?v=tRpHkcjgkdw"
    ],
    "Diabetic Wounds": [
        "https://www.youtube.com/watch?v=iPgnjJICzos",
        "https://www.youtube.com/watch?v=IHlV2M-8a2Q",
        "https://www.youtube.com/watch?v=N8M4qj6w82c"
    ],
    "Pressure Wounds": [
        "https://www.youtube.com/watch?v=BiqLzQ6nHWM",
        "https://www.youtube.com/watch?v=ynP1Ocooh3E"
    ],
    "Surgical Wounds": [
        "https://www.youtube.com/watch?v=R4DutP4ya9w",
        "https://www.youtube.com/watch?v=tRpHkcjgkdw"
    ],
    "Bruises": [
        "https://www.youtube.com/watch?v=4E_tScuwOYo",
        "https://www.youtube.com/watch?v=0uZpPpxM1aE"
    ],
    "Venous Wounds": [
        "https://www.youtube.com/watch?v=4ZUXy1kJqDg",
        "https://www.youtube.com/watch?v=nL-H3RSlFOQ"
    ],
}
def get_first_aid_videos(label: str):
    return first_aid_videos.get(label, [])

# ================== OTC / Rx MODULE (tối giản) ==================
OTC_ACTIVES = {
    "paracetamol": {"alias": ["acetaminophen"], "note": "OTC liều thường 325–500mg; cảnh báo quá liều"},
    "ibuprofen": {"note": "OTC liều thấp người lớn; cân nhắc dạ dày/ thai kỳ"},
    "loratadine": {"note": "kháng histamine thế hệ 2 (ít buồn ngủ)"},
    "cetirizine": {"note": "kháng histamine; có thể gây buồn ngủ nhẹ"},
    "fexofenadine": {},
    "chlorpheniramine": {"note": "kháng H1 đời cũ; gây buồn ngủ rõ"},
    "dextromethorphan": {"note": "ức chế ho; tránh lạm dụng"},
    "guaifenesin": {"note": "long đờm"},
    "ors": {"alias": ["oresol", "oral rehydration salts"], "note": "bù nước điện giải"},
    "simethicone": {"note": "đầy hơi, chướng bụng"},
    "aluminum hydroxide": {"alias": ["aluminium hydroxide"]},
    "magnesium hydroxide": {},
    "activated charcoal": {"alias": ["than hoat", "than hoạt"]},
    "sodium cromoglicate": {"alias": ["sodium cromoglycate"], "note": "viêm kết mạc/viêm mũi dị ứng nhẹ"},
    "oxymetazoline": {"note": "xịt mũi: chỉ dùng ngắn ngày (≤3 ngày)"},
}
ALIASES = {}
for k, v in list(OTC_ACTIVES.items()):
    for a in v.get("alias", []):
        ALIASES[a.lower()] = k

def _norm(s: str):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 +./-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize_acts(s: str):
    s = _norm(s)
    parts = re.split(r"[+,/;]| with | and ", s)
    acts = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p in ALIASES:
            p = ALIASES[p]
        acts.append(p)
    return [a for a in acts if a]

def classify_rx_otc(input_text: str):
    acts = _tokenize_acts(input_text)
    if not acts:
        return {"status": "unknown", "hits": [], "reason": "Không nhận diện được hoạt chất."}
    hits, all_otc = [], True
    for a in acts:
        matched = None
        for k in OTC_ACTIVES.keys():
            if k in a:
                matched = k
                break
        if matched:
            hits.append((a, "OTC", OTC_ACTIVES[matched].get("note", "")))
        else:
            all_otc = False
            hits.append((a, "Rx", "Không nằm trong danh sách OTC rút gọn (local)"))
    status = "OTC" if all_otc else "Rx/Không chắc"
    return {"status": status, "hits": hits, "reason": "Dựa trên danh sách OTC local (cần xác minh khi không chắc)."}

# ================== SESSION STATE ==================
if "wound_records" not in st.session_state:
    st.session_state.wound_records = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# lưu kết quả dự đoán gần nhất (label, info, confidence)
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "user_info" not in st.session_state:
    st.session_state.user_info = {}

# ================== TABS ==================
tabs = st.tabs([
    "🩺 Ảnh vết thương",
    "💊 Thuốc OTC / Kê đơn",
    "🤖 Bác sĩ AI",
    "📊 Thống kê",
    "🌦️ Thời tiết & Gợi ý chăm sóc",

])
tab1, tab2, tab3, tab4 ,tab5= tabs

# ================== TAB 1: ẢNH VẾT THƯƠNG ==================
with tab1:
    st.title("🩺 Chương trình phân loại vết thương")
    st.write("Upload ảnh chụp để chương trình dự đoán loại vết thương & gợi ý phương án sơ cứu.")
    st.divider()

    uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã upload", use_container_width=True)

        # Dự đoán
        label, confidence, scores = predict_image(image)
        info = wound_info[label]
        # Lưu kết quả dự đoán vào session để các tab khác có thể dùng
        st.session_state.last_prediction = {
            "label": label,
            "info": info,
            "confidence": confidence,
            "scores": scores
        }

        left, right = st.columns([1, 1])

        with left:
            st.markdown(f"### ✅ Kết quả: **{info['ten_viet']} ({label})**")
            st.markdown(f"**Độ tin cậy:** {confidence*100:.2f}%")

            # ====== THÔNG TIN NGUY CƠ & KHI NÀO ĐI KHÁM ======
            muc_do = info.get("muc_do", "Chưa xác định")
            st.markdown(f"### 🩹 **Mức độ nguy cơ:** {muc_do}")
            if any(k in muc_do.lower() for k in ["vừa/nặng", "nặng"]):
                st.error("🚨 Khuyến nghị: Đến cơ sở y tế để được xử trí chuyên sâu.")
            else:
                st.success("✅ Có thể sơ cứu tại nhà và theo dõi thêm 1–2 ngày.")

            with st.expander("⚠️ Khi nào cần đi khám ngay?", expanded=False):
                st.write(info.get("canh_bao", "Theo dõi triệu chứng bất thường và đi khám khi cần."))

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
                idx = 0 if len(vids) == 1 else st.selectbox(
                    "Chọn video:", range(len(vids)), format_func=lambda x: f"Video {x+1}"
                )
                render_youtube(vids[idx], height=460)

        st.divider()

        # --- 📝 NHẬT KÝ ---
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
            notes = st.text_input("Ghi chú khác", placeholder="Dị ứng thuốc, bệnh nền...")

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
            st.download_button("⬇️ Tải toàn bộ nhật ký (CSV)", data=csv_bytes,
                               file_name="wound_log.csv", mime="text/csv")
    else:
        st.info("Hãy chọn một ảnh để bắt đầu chẩn đoán và ghi nhật ký.")

# ================== TAB 2: GỢI Ý THUỐC + MUA ONLINE (Long Châu) + Phân loại ==================
# ================== TAB 2: GỢI Ý THUỐC + MUA ONLINE (Long Châu) + Phân loại ==================
with tab2:
    st.header("💊 Gợi ý & Phân loại thuốc/thiết bị")

    # Danh sách gợi ý thuốc theo loại vết thương
    # Danh sách gợi ý thuốc theo loại vết thương
    DRUG_SUGGESTIONS = {
        # 3 loại đã có
        "vết trầy xước": [ # Khớp với Abrasions
            {"item": "Betadine 10%", "why": "Sát khuẩn vết thương", "how": "Rửa nhẹ 2–3 lần/ngày", "note": "Tránh dùng quá nhiều"},
            {"item": "Gạc vô trùng", "why": "Bảo vệ vùng tổn thương", "how": "Thay mỗi 12h", "note": "Giữ khô"}
        ],
        "bỏng nhẹ": [ # Khớp với Burns (nếu là bỏng nhẹ)
            {"item": "Silvirin cream 1%", "why": "Chống nhiễm khuẩn, làm dịu da", "how": "Bôi mỏng 1–2 lần/ngày", "note": "Chỉ dùng ngoài da"},
            {"item": "NaCl 0.9%", "why": "Làm sạch vết bỏng", "how": "Rửa nhẹ trước khi bôi thuốc", "note": ""}
        ],
        "vết cắt nhẹ": [ # Khớp với Cut (nếu là cắt nhẹ)
            {"item": "Oxy già 3%", "why": "Rửa vết thương ban đầu", "how": "Chỉ dùng 1 lần đầu tiên", "note": "Không lạm dụng"},
            {"item": "Mỡ kháng sinh (Fucidin, Tetracycline)", "why": "Ngừa nhiễm trùng", "how": "Bôi mỏng ngày 2 lần", "note": ""}
        ],

        # 7 loại cần bổ sung (ví dụ)
        # Bạn cần tự định nghĩa nội dung gợi ý cho các loại này
        
        "vết bầm": [ # Khớp với Bruises
            {"item": "Chườm lạnh", "why": "Giảm sưng, co mạch", "how": "Chườm 10-15 phút, vài lần/ngày (24h đầu)", "note": "Không chườm đá trực tiếp lên da"},
            {"item": "Chườm nóng", "why": "Tan máu bầm", "how": "Sau 24-48h", "note": ""}
        ],
        "loét tiểu đường": [ # Khớp với Diabetic Wounds
            {"item": "Gạc chuyên dụng (hydrocolloid...)", "why": "Duy trì môi trường ẩm, bảo vệ", "how": "Theo chỉ định BS", "note": "Cần kiểm soát đường huyết!"},
            {"item": "Dung dịch sát khuẩn (Betadine/NaCl)", "why": "Làm sạch", "how": "Rửa nhẹ nhàng", "note": "Tuyệt đối tuân thủ chỉ định y tế"}
        ],
        "vết rách": [ # Khớp với Laceration
            {"item": "Gạc ép cầm máu", "why": "Cầm máu ban đầu", "how": "Ép chặt và giữ", "note": "Vết rách sâu/rộng cần đi khâu"},
            {"item": "Băng dán (nếu nông)", "why": "Bảo vệ", "how": "Sau khi sát khuẩn", "note": ""}
        ],
        "da lành": [ # Khớp với Normal
            {"item": "Kem dưỡng ẩm", "why": "Duy trì sức khỏe da", "how": "Hàng ngày", "note": "Không cần can thiệp y tế"}
        ],
        "loét tì đè": [ # Khớp với Pressure Wounds
            {"item": "Gạc xốp (foam dressing)", "why": "Giảm áp lực, hút dịch", "how": "Theo chỉ định", "note": "Quan trọng nhất là thay đổi tư thế thường xuyên"},
            {"item": "Đệm chống loét", "why": "Phân tán áp lực", "how": "Sử dụng cho bệnh nhân", "note": ""}
        ],
        "vết mổ": [ # Khớp với Surgical Wounds
            {"item": "Gạc vô trùng", "why": "Bảo vệ vết khâu", "how": "Thay băng theo chỉ định", "note": "Giữ khô tuyệt đối (trừ khi được phép)"},
            {"item": "Dung dịch Povidine-Iodine", "why": "Sát khuẩn khi thay băng", "how": "Theo hướng dẫn của BS", "note": ""}
        ],
        "loét tĩnh mạch": [ # Khớp với Venous Wounds
            {"item": "Băng ép (vớ y khoa)", "why": "Tăng cường lưu thông máu về tim", "how": "Đeo hàng ngày", "note": "Cần thăm khám chuyên khoa"},
            {"item": "Gạc tẩm bạc (nếu nhiễm trùng)", "why": "Diệt khuẩn", "how": "Theo chỉ định", "note": ""}
        ]
    }

    # Hàm chuẩn hóa chuỗi
    def normalize_str(s: str):
        s = (s or "").lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Lấy loại vết thương từ dự đoán hoặc chọn thủ công
    last = st.session_state.get("last_prediction")
    mapped_key = None
    pred_conf = None
    if last is not None:
        raw = last.get("info", {}).get("ten_viet", "")
        pred_conf = float(last.get("confidence", 0)) if last.get("confidence") is not None else None
        norm_pred = normalize_str(raw)
        norm_map = {normalize_str(k): k for k in DRUG_SUGGESTIONS.keys()}

        if norm_pred in norm_map:
            mapped_key = norm_map[norm_pred]
        else:
            for nk, orig in norm_map.items():
                if nk in norm_pred or norm_pred in nk:
                    mapped_key = orig
                    break
            if mapped_key is None:
                matches = difflib.get_close_matches(norm_pred, list(norm_map.keys()), n=1, cutoff=0.6)
                if matches:
                    mapped_key = norm_map[matches[0]]

    if mapped_key:
        src = f"AI ({pred_conf*100:.1f}% tin cậy)" if pred_conf is not None else "AI"
        st.caption(f"Loại vết thương: {mapped_key} (Nguồn: {src})")
        wound_type = mapped_key
    else:
        wound_type = st.selectbox("Chọn loại vết thương:", list(DRUG_SUGGESTIONS.keys()))

    # Hiển thị gợi ý thuốc
    found = False
    for key, suggestions in DRUG_SUGGESTIONS.items():
        if key == wound_type:
            found = True
            st.subheader("Gợi ý sử dụng:")
            for s in suggestions:
                with st.container():
                    st.markdown(f"**{s['item']}**: {s['why']} | *{s['how']}*")
                    if s.get("note"):
                        st.caption(f"⚠️ Lưu ý: {s['note']}")
            break

    if not found:
        st.info("Hiện chưa có gợi ý cụ thể cho loại vết thương này.")

    st.caption("Thông tin mang tính tham khảo. Hãy tham khảo ý kiến bác sĩ/dược sĩ trước khi sử dụng.")
    st.divider()

    # Tra cứu cơ sở y tế
    with st.container():
        st.subheader("🏥 Cơ sở y tế gần bạn")
        city = (st.session_state.get("user_info", {}).get("Tỉnh/Thành phố", "") or "").strip()
        if city:
            st.write(f"🔎 Tra cứu bệnh viện gần: **{city}**")
            # Sửa lỗi URL Google Maps, tránh các domain không an toàn
            # Chuyển https://www.google.com/maps/search/bệnh+viện+gần+...
            # thành https://www.google.com/maps/search/?api=1&query=...
            query_encoded = _u.quote(f"bệnh viện tại {city}")
            gmap_url = f"https://www.google.com/maps/search/?api=1&query={query_encoded}"
            st.markdown(f"[📍 Mở Google Maps]({gmap_url})")
        else:
            st.warning("⚠️ Vui lòng cập nhật Tỉnh/Thành phố trong hồ sơ.")
        st.markdown("[🌐 Tra cứu thuốc (DAV)](https://dav.gov.vn/tra-cuu-thuoc.html)")
# ================== TAB 3: BÁC SĨ AI (CHAT) ==================
with tab3:
    st.header("🤖 Tư vấn AI - Bác sĩ ảo")
    # Hiển thị lịch sử
    for chat in st.session_state.chat_history:
        role = "🧑‍⚕️ Bác sĩ AI" if chat["role"] == "assistant" else "🧍‍♂️ Bạn"
        with st.chat_message(chat["role"]):
            st.markdown(f"**{role}:** {chat['content']}")

    # Nhập câu hỏi
    user_input = st.chat_input("Nhập câu hỏi của bạn (vd: Vết thương này có cần băng lại không?)")
    if user_input:
        st.chat_message("user").markdown(f"**Bạn:** {user_input}")
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma:2b", "prompt": user_input, "stream": False},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            ai_text = data.get("response", "").strip()

            st.chat_message("assistant").markdown(f"**🧑‍⚕️ Bác sĩ AI:** {ai_text}")
            st.session_state.chat_history.append({"role": "assistant", "content": ai_text})

        except requests.exceptions.RequestException as e:
            st.error(f"🚫 Lỗi khi kết nối Ollama: {e}")
            st.info("⚙️ Kiểm tra: đã chạy `ollama serve` và `ollama pull gemma:2b` chưa?")

# ================== TAB 4: 📊 THỐNG KÊ & BÁO CÁO ==================
with tab4:
    st.header("📊 Thống kê & Báo cáo")

    if not st.session_state.wound_records:
        st.info("Chưa có ca nào trong nhật ký. Hãy chạy phân loại ở Tab 1 để có dữ liệu thống kê.")
    else:
        df = pd.DataFrame(st.session_state.wound_records).copy()

        # Chuẩn hoá thời gian & ngày
        try:
            df["Thời gian"] = pd.to_datetime(df["Thời gian"])
        except Exception:
            pass
        if "Thời gian" in df.columns:
            df["Ngày"] = df["Thời gian"].dt.date
        else:
            df["Ngày"] = date.today()

        # --- Bộ lọc ---
        colf1, colf2, colf3 = st.columns([1,1,1])
        with colf1:
            min_day = min(df["Ngày"])
            max_day = max(df["Ngày"])
            d_from, d_to = st.date_input("Khoảng ngày", value=(min_day, max_day))
        with colf2:
            labels = sorted(df["Chẩn đoán (AI)"].unique().tolist()) if "Chẩn đoán (AI)" in df.columns else []
            pick_labels = st.multiselect("Lọc theo nhãn", options=labels, default=labels)
        with colf3:
            severities = sorted(df["Mức độ"].unique().tolist()) if "Mức độ" in df.columns else []
            pick_sev = st.multiselect("Lọc theo mức độ", options=severities, default=severities)

        # Áp bộ lọc
        mask = (df["Ngày"] >= d_from) & (df["Ngày"] <= d_to)
        if pick_labels:
            mask &= df["Chẩn đoán (AI)"].isin(pick_labels)
        if pick_sev:
            mask &= df["Mức độ"].isin(pick_sev)
        dff = df[mask].copy()

        st.markdown(f"**Số ca sau lọc:** {len(dff)}")

        # --- Biểu đồ ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🗓️ Số ca theo ngày")
            daily = dff.groupby("Ngày").size().rename("Số ca")
            if len(daily) == 0:
                st.info("Không có dữ liệu trong khoảng lọc.")
            else:
                st.line_chart(daily)

        with col2:
            st.subheader("🏷️ Phân bố theo nhãn")
            if "Chẩn đoán (AI)" in dff.columns:
                by_label = dff["Chẩn đoán (AI)"].value_counts()
                if len(by_label) == 0:
                    st.info("Không có dữ liệu.")
                else:
                    st.bar_chart(by_label)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("🩹 Phân bố theo mức độ")
            if "Mức độ" in dff.columns:
                by_sev = dff["Mức độ"].value_counts()
                if len(by_sev) == 0:
                    st.info("Không có dữ liệu.")
                else:
                    st.bar_chart(by_sev)

        with col4:
            st.subheader("🎯 Độ tin cậy (trung bình)")
            if "Độ tin cậy (%)" in dff.columns and not dff["Độ tin cậy (%)"].empty:
                avg_acc = round(float(dff["Độ tin cậy (%)"].mean()), 2)
                st.metric("Độ tin cậy trung bình (%)", avg_acc)
            else:
                st.info("Chưa có trường 'Độ tin cậy (%)'.")

        st.divider()
        st.subheader("📝 Ca gần nhất (10 dòng)")
        st.dataframe(
            dff.sort_values("Thời gian", ascending=False).head(10),
            use_container_width=True, height=260
        )

        # --- Tải về CSV ---
        csv_all = dff.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Tải dữ liệu đã lọc (CSV)", data=csv_all,
         file_name="thong_ke_wound_records.csv", mime="text/csv")
# ================== TAB 5: THỜI TIẾT & GỢI Ý CHĂM SÓC ==================
with tab5:
    st.header("🌦️ Thời tiết & Gợi ý chăm sóc sức khỏe")

    # Lấy sẵn từ sidebar nếu có
    city_tt = (st.session_state.user_info.get("Tỉnh/Thành phố", "") or "").strip()
    city_tt = st.text_input(
        "Nhập Tỉnh/Thành phố để xem thời tiết",
        value=city_tt,
        placeholder="VD: Hà Nội, Đà Nẵng, TP. Hồ Chí Minh"
    )

    if city_tt:
        with st.spinner(f"Đang cập nhật thời tiết tại {city_tt}..."):
            try:
                q_city = _u.quote(city_tt)
                url = f"https://wttr.in/{q_city}?format=j1"
                data = requests.get(url, timeout=10).json()

                cur = data["current_condition"][0]
                temp = float(cur["temp_C"])
                feels_like = float(cur["FeelsLikeC"])
                humidity = int(cur["humidity"])
                desc = cur["weatherDesc"][0]["value"]

                st.markdown(f"### 📍 {city_tt}")
                colA, colB = st.columns(2)
                with colA:
                    st.metric("🌡️ Nhiệt độ (°C)", f"{temp:.1f}", delta=f"Cảm giác: {feels_like:.1f}°C")
                with colB:
                    st.metric("💧 Độ ẩm (%)", f"{humidity}%")
                st.write(f"**Mô tả:** {desc}")

                # Gợi ý chăm sóc rất cơ bản
                if "mưa" in desc.lower() or "rain" in desc.lower():
                    suggestion = "Mang áo mưa/ô, giày chống trượt; lau khô người sớm."
                elif temp <= 20:
                    suggestion = "Mặc ấm, uống nước ấm; tránh gió lạnh buổi sáng."
                elif temp >= 33:
                    suggestion = "Uống đủ nước, tránh nắng gắt; bôi chống nắng khi ra ngoài."
                elif humidity >= 85:
                    suggestion = "Độ ẩm cao: giữ khô thoáng, phơi đồ kỹ, chú ý da/nấm."
                else:
                    suggestion = "Thời tiết dễ chịu; duy trì ăn ngủ điều độ."

                st.success(f"💡 Gợi ý chăm sóc: {suggestion}")

                # Đồ dùng nên chuẩn bị (ngắn gọn)
                items = []
                if "mưa" in desc.lower() or "rain" in desc.lower():
                    items += ["Áo mưa/ô", "Khăn lau khô"]
                if temp <= 20:
                    items += ["Áo khoác ấm", "Khẩu trang"]
                if temp >= 33:
                    items += ["Mũ/nón", "Kem chống nắng", "Bình nước"]
                if not items:
                    items = ["Khẩu trang", "Nước uống"]

                st.markdown("**🧳 Gợi ý mang theo:** " + ", ".join(items))
                st.caption("Nguồn: wttr.in (thời gian thực)")

            except Exception as e:
                st.error(f"Không lấy được dữ liệu thời tiết: {e}")
    else:
        st.info("Nhập Tỉnh/Thành phố để xem thời tiết và gợi ý chăm sóc.")
