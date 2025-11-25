import os
import re
import io
import base64
import requests
import pandas as pd
from PIL import Image
from flask_cors import CORS
from dotenv import load_dotenv
from collections import defaultdict
import google.generativeai as genai
from flask import Flask, request, jsonify
from datetime import date, timedelta, datetime 

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==================== Gemini 2.0 Flash ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY مش موجود! روح https://aistudio.google.com/app/apikey وخد واحد ببلاش")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel(
    'gemini-2.0-flash',
    generation_config={"temperature": 0.8, "max_output_tokens": 2048}
)

# ==================== CSV Data ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'products.csv')
CSV_DATA = pd.read_csv(csv_path)

# ==================== IP & Location & Weather ====================
IPV4_PRIVATE = re.compile(r'^(127\.0\.0\.1|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)')

def is_private_ip(ip: str) -> bool:
    return bool(IPV4_PRIVATE.match(ip))

def get_user_ip() -> str:
    headers = ["CF-Connecting-IP", "True-Client-IP", "X-Real-IP", "X-Forwarded-For", "X-Client-IP", "Forwarded"]
    for h in headers:
        val = request.headers.get(h)
        if val:
            ips = [i.strip() for i in val.replace('"', '').split(",")]
            for ip in ips:
                if ip and not is_private_ip(ip):
                    return ip
    return request.remote_addr or "127.0.0.1"

def get_location(ip: str):
    try:
        r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=8)
        r.raise_for_status()
        d = r.json()
        if d.get("error") or not d.get("city") or not d.get("latitude") or not d.get("longitude"):
            raise ValueError("بيانات ناقصة")
        return {"city": d.get("city"), "lat": d.get("latitude"), "lon": d.get("longitude")}
    except:
        try:
            r = requests.get(f"https://ipwho.is/{ip}", timeout=8)
            r.raise_for_status()
            d = r.json()
            if not d.get("city") or not d.get("latitude") or not d.get("longitude"):
                return None
            return {"city": d.get("city"), "lat": d.get("latitude"), "lon": d.get("longitude")}
        except Exception as e:
            print(f"Location error: {e}")
            return None

def fetch_weather(lat, lon):
    start = date.today()
    end = start + timedelta(days=13)
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()["daily"]
    except Exception as e:
        print(f"Weather error: {e}")
        return None

# ==================== Memory ====================
conversation_history = defaultdict(list)
user_disabled_products = set()
user_mood_asked = set()

def gemini_chat(user_message, image_b64=None):
    try:
        user_ip = get_user_ip()
        today_temp = round((weather_data["temperature_2m_max"][0] + weather_data["temperature_2m_min"][0]) / 2, 1)

        products_text = "المنتجات المتاحة (ترشح من دول بس وما تطلعش حاجة برا القايمة):\n"
        for _, row in CSV_DATA.iterrows():
            name = str(row.get('name') or row.get('اسم المنتج') or row.get('product_name') or row.iloc[0]).strip()
            price = row.get('price') or row.get('السعر') or row.iloc[1]
            cat = str(row.get('category') or row.get('الكاتيجوري') or row.get('القسم') or row.iloc[2] if len(row) > 2 else "غير محدد").strip()
            id_ = row.get('id') or row.get('product_id') or row.iloc[3] if len(row) > 3 else "unknown"
            products_text += f"• {name} | السعر: {price} جنيه | الكاتيجوري: {cat} | اللينك: https://afaq-stores.com/product-details/{id_}\n"

        show_products = user_ip not in user_disabled_products

        mood_prompt = ""
        if user_ip not in user_mood_asked and len(conversation_history[user_ip]) == 0:
            mood_prompt = "إبدأ المحادثة بسؤاله: \"مزاجك إيه النهاردة؟ عايز لبس شيك ولا كاجوال؟\" وبعدين كمل عادي."

        full_message = f"""
        أنت شاب مصري اسمه "عبدالله" (أو عبدو)، صاحب محل لبس شيك في {city}، بتتكلم عامية مصرية خفيفة وطبيعية جدًا، مرح، ودود، وبتفهم في الموضة.

        الجو في {city} النهاردة: {today_temp}°C

        لو اليوزر قال أي حاجة زي "مش عايز منتجات" أو "بس بشوف" أو "مش هاشتري دلوقتي" → متعرضش أي منتج نهائي لحد ما يقول صراحة "رشحلي" أو "عايز اشتري".

        المنتجات المتاحة (ترشح من دول بس لو مسموح):
        {products_text if show_products else "اليوزر حاليًا مش عايز يشوف منتجات"}

        المحادثة السابقة:
        {chr(10).join([text for role, text in conversation_history[user_ip][-10:]])}

        اليوزر بيقول دلوقتي: {user_message or "فيه صورة مرفوعة"}

        ≫≫ تعليمات مهمة جدًا ≪≪
        {mood_prompt}

        - لو رفع صورة هو بيسأل عن منتج → قوله: "ثانية بس أشوف الصورة... الله ينور! طب لو عندك صورة تانية للي في دماغك ابعتها وأنا أجيبلك زيها بالظبط"
        - لو قال "غالي" أو "مش هينفع" → اعمل فصال مصري: "طب قوللي ميزانيتك كام وأنا أجيبلك حاجة تليق بيك" أو "خلاص لو هتاخد اتنين نعملك خصم حلو"
        - لو محتار بين اتنين → قوله: "طب أنا لو مكانك هاخد كذا، لأنه بيلبس أكتر وبيخسس كمان"
        - لو سكت كتير → ردود زي: "لسه معايا؟" أو "غمزت ولا لسه بتفكر؟"
        - لو أول مرة يكلمني → اسأله عن مزاجه.
        - ماتقولش "أنت:" أبدًا، خليك طبيعي 100%.
        - متعترفش إنك بوت أبدًا.
        - ردودك مرحة، خفيفة، ومصرية جدًا.
        - متردش بي اي ايموجي نهائي.

        رد دلوقتي بطريقة طبيعية كأنك بتكلم زبون دخل المحل.
        """.strip()

        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes))
            response = MODEL.generate_content([full_message, img])
        else:
            response = MODEL.generate_content(full_message)

        reply = response.text.strip()
        return reply

    except Exception as e:
        print(f"Gemini Error: {e}")
        return "ثواني بس وهرجعلك تاني!"

@app.route("/")
def home():
    return "PureSoft AI Backend شغال 100% مع Gemini 2.0 Flash"

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        user_ip = get_user_ip()
        location = get_location(user_ip)
        if not location:
            return jsonify({"error": "مش عارف أحدد مكانك"}), 400

        global city, weather_data
        city = location["city"]
        weather_data = fetch_weather(location["lat"], location["lon"])
        if not weather_data:
            return jsonify({"error": "مشكلة في جلب الطقس"}), 500

        user_message = request.form.get("message", "").strip()
        image_file = request.files.get("image")
        image_b64 = None
        if image_file:
            if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                return jsonify({"error": "نوع الصورة مش مدعوم"}), 400
            if image_file.content_length > 5 * 1024 * 1024:
                return jsonify({"error": "الصورة كبيرة أوي"}), 400
            image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

        if not user_message and not image_b64:
            return jsonify({"error": "لازم تبعت رسالة أو صورة"}), 400

        lower_msg = user_message.lower()
        if any(phrase in lower_msg for phrase in ["مش عايز", "مفيش حاجة", "مش هاشتري", "بس بشوف", "مش عايز منتجات"]):
            user_disabled_products.add(user_ip)
        if any(phrase in lower_msg for phrase in ["رشحلي", "عايز اشتري", "وريني", "عايز حاجة", "ايه عندك"]):
            user_disabled_products.discard(user_ip)
            user_mood_asked.add(user_ip)

        reply = gemini_chat(user_message, image_b64)

        conversation_history[user_ip].append(("user", user_message or "[صورة]"))
        conversation_history[user_ip].append(("assistant", reply))
        if len(conversation_history[user_ip]) > 30:
            conversation_history[user_ip] = conversation_history[user_ip][-30:]

        return jsonify({
            "reply": reply,
            "city": city
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "فيه مشكلة، جرب تاني"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

