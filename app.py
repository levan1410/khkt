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
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import urllib.parse as _u

# ============ PAGE CONFIG ============
st.set_page_config(page_title="AI Sơ cứu & Dị ứng", layout="wide")

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

# ================== TABS ==================
tabs = st.tabs([
    "🩺 Ảnh vết thương",
    "💊 Thuốc OTC / Kê đơn",
    "🤖 Bác sĩ AI",
    "📊 Thống kê",
])
tab1, tab2, tab3, tab4 = tabs

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
with tab2:
    st.header("💊 Gợi ý mua & Phân loại thuốc / thiết bị")
    SUGGEST_BY_NEED = {
        "Dị ứng mũi/da nhẹ": [
            {"item":"Loratadine 10 mg", "why":"dị ứng theo mùa/viêm mũi dị ứng", "how":"1 viên/ngày", "note":"ít gây buồn ngủ", "lc_link":"https://nhathuoclongchau.com.vn/tim-kiem?q=loratadine%2010mg"},
            {"item":"Cetirizine 10 mg", "why":"dị ứng/ ngứa da", "how":"1 viên/ngày", "note":"tối có thể buồn ngủ", "lc_link":"https://nhathuoclongchau.com.vn/tim-kiem?q=cetirizine%2010mg"},
            {"item":"Sodium cromoglicate 2% (nhỏ mắt)", "why":"ngứa/đỏ mắt dị ứng", "how":"1–2 giọt", "note":"không thay kháng sinh", "lc_link":"https://nhathuoclongchau.com.vn/tim-kiem?q=sodium%20cromoglicate%202%25"},
            {"item":"Nước muối NaCl 0.9%", "why":"rửa mũi/mắt", "how":"xịt/nhỏ theo nhu cầu", "note":"", "lc_link":"https://nhathuoclongchau.com.vn/tim-kiem?q=nacl%200.9%25"}
        ],
        "Đau/ sốt nhẹ": [
            {"item":"Paracetamol 500 mg", "why":"giảm đau–hạ sốt", "how":"1 viên mỗi 6–8 giờ", "note":"tránh rượu/thuốc khác", "lc_link":"https://nhathuoclongchau.com.vn/tim-kiem?q=paracetamol%20500mg"},
            {"item":"Ibuprofen 200 mg", "why":"đau viêm nhẹ", "how":"200–400 mg mỗi 6–8 giờ", "note":"dạ dày/ thai kỳ", "lc_link":"https://nhathuoclongchau.com.vn/tim-kiem?q=ibuprofen%20200mg"}
        ],
    }

    need = st.selectbox("Chọn nhu cầu:", list(SUGGEST_BY_NEED.keys()))
    if need:
        items = SUGGEST_BY_NEED[need]
        for it in items:
            st.write(f"• **{it['item']}** — {it['why']}"
                     + (f" | Cách dùng: _{it.get('how','')}_ " if it.get('how') else "")
                     + (f" | ⚠️ {it.get('note','')}" if it.get('note') else ""))
            if it.get("lc_link"):
                st.markdown(f"[🛒 Mua tại Long Châu]({it['lc_link']})")

    st.caption("Lưu ý: Thông tin tham khảo — hãy hỏi dược sĩ để được tư vấn phù hợp.")
    st.divider()

    # --- Phân loại OTC/Rx gọn ---
    st.subheader("🔎 Phân loại hoạt chất: OTC hay Rx")
    drug_q = st.text_input("Nhập tên **hoạt chất** hoặc **kết hợp** (vd: paracetamol 500mg + loratadine)")
    col_rx1, col_rx2 = st.columns([1,1])

    with col_rx1:
        if st.button("Phân loại"):
            if not drug_q.strip():
                st.warning("Vui lòng nhập tên thuốc/hoạt chất.")
            else:
                res = classify_rx_otc(drug_q)
                st.subheader(f"Kết quả: {res['status']}")
                if res["hits"]:
                    for a, lab, note in res["hits"]:
                        st.write(f"- **{a}** → **{lab}**")
                if res["status"] == "OTC":
                    lc_q = _u.quote_plus(drug_q)
                    st.markdown(f"[🛒 Mua online tại Long Châu](https://nhathuoclongchau.com.vn/tim-kiem?q={lc_q})")
                else:
                    st.info("⛑️ Thuốc kê đơn/không chắc: hỏi dược sĩ hoặc bác sĩ trước khi mua/dùng.")

    with col_rx2:
        st.info("Không chắc liều/đối tượng dùng? Hãy hỏi **dược sĩ** tại nhà thuốc gần nhất.")
        st.markdown("[🌐 Tra cứu chính thức (DAV)](https://dav.gov.vn/tra-cuu-thuoc.html)")

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
                "https://belle-buyer-refuse-performing.trycloudflare.com",
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
