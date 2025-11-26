import os
import io
import base64
import requests
import pandas as pd
from PIL import Image
import google.generativeai as genai
from flask_cors import CORS
from dotenv import load_dotenv
from collections import defaultdict
from flask import Flask, request, jsonify

# ←←← إضافة جديدة (سطرين بس)
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==================== Gemini Setup ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY مش موجود في .env")

genai.configure(api_key=GEMINI_API_KEY)

# ←←← تعديل بسيط جدًا: أضفنا safety_settings فقط
safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

MODEL = genai.GenerativeModel(
    'gemini-2.0-flash',
    generation_config={"temperature": 0.85, "max_output_tokens": 2048},
    safety_settings=safety_settings        # ←←← السطر الوحيد اللي زاد هنا
)

# ==================== Load CSV ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'products.csv')
CSV_DATA = pd.read_csv(csv_path)

# ==================== IP & Weather ====================
def get_user_ip():
    headers = ["CF-Connecting-IP", "True-Client-IP", "X-Real-IP", "X-Forwarded-For"]
    for h in headers:
        val = request.headers.get(h)
        if val:
            ips = [i.strip() for i in val.replace('"', '').split(",")]
            for ip in ips:
                if ip and not ip.startswith(('127.', '10.', '172.', '192.168.')):
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
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
        r = requests.get(url, timeout=10)
        data = r.json()["daily"]
        temp = round((data["temperature_2m_max"][0] + data["temperature_2m_min"][0]) / 2, 1)
        return temp
    except Exception as e:
        print(f"Weather error: {e}")
        return None

# ==================== Memory ====================
conversation_history = defaultdict(list)

# ==================== Gemini Chat ====================
def gemini_chat(user_message="", image_b64=None):
    try:
        user_ip = get_user_ip()
        location = get_location(user_ip)
        city = location["city"]
        today_temp = fetch_weather(location["lat"], location["lon"])

        products_text = "المنتجات المتاحة (ممنوع تغيير ولا حرف في الاسم أبدًا):\n"
        for _, row in CSV_DATA.iterrows():
            name = str(row['product_name_ar']).strip()
            price = float(row['sell_price'])
            cat = str(row['category']).strip()
            pid = str(row['product_id'])
            products_text += f"• {name} | السعر: {price} جنيه | الكاتيجوري: {cat} | اللينك: https://afaq-stores.com/product-details/{pid}\n"

        full_message = f"""
أنت شاب مصري بتتكلم عامية مصرية طبيعية وودودة جدًا، بتعرف تحلل صور وبتفهم في الموضة والعناية الشخصية كويس.

الجو في {city} النهاردة: {today_temp}°C

المنتجات اللي عندك (لازم تنسخ الاسم بالحرف من غير أي تغيير أو تلخيص أو اختراع نهائي):

{products_text}

المحادثة السابقة:
{chr(10).join([t for _, t in conversation_history[user_ip][-10:]])}

اليوزر بيقول: {user_message or "فيه صورة مرفوعة"}

≫≫ قواعد حديدية – ممنوع تخالفها أبدًا ≪≪

لو اليوزر عايز لبس أو عناية أو رفع صورة لمنتج → لازم ترشح منتجات من القايمة فوق بالتنسيق ده بالظبط:

تيشيرت قطن سادة ابيض
السعر: 130 جنيه
الكاتيجوري: لبس صيفي
اللينك: https://afaq-stores.com/product-details/1019

سكارف كشمير طويل
السعر: 290 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1014

جاكيت جلد اسود تقيل مبطن فرو
السعر: 720 جنيه
الكاتيجوري: لبس شتوي
اللينك: https://afaq-stores.com/product-details/1001

ممنوع منعًا باتًا:
- تكتب "لبس صيفي" أو "جاكيت" أو "تيشيرت" كاسم منتج
- تختصر أو تعيد صياغة الاسم
- تخترع اسم جديد أبدًا

لو اليوزر رفع صورة عادية أو سأل حاجة مش عن منتجات → حلل الصورة ورد طبيعي من غير منتجات.
لو رفع صورة + كلام عن لبس أو عناية → حلل الصورة الأول وبعدين رشح بالتنسيق أعلاه.

- ردك عامية مصرية 100%
- متستخدمش إيموجي نهائي
- متعترفش إنك بوت أبدًا
- لو فيه صورة → ابدأ بـ "ثانية بس أشوف الصورة..."

رد دلوقتي.
""".strip()

        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes))
            response = MODEL.generate_content([full_message, img])
        else:
            response = MODEL.generate_content(full_message)

        if not getattr(response, "candidates", None):
            raise ValueError("تم حظر الرد من جوجل (candidates empty)")

        reply = response.text.strip()
        return reply

    except Exception as e:
        print(f"Gemini Error: {e}")
        return "ثواني بس وأرجعلك…"

# ==================== Routes ====================
@app.route("/")
def home():
    return "البوت شغال 100%"

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        user_ip = get_user_ip()
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

        reply = gemini_chat(user_message, image_b64)

        conversation_history[user_ip].append(("user", user_message or "[صورة]"))
        conversation_history[user_ip].append(("assistant", reply))
        if len(conversation_history[user_ip]) > 30:
            conversation_history[user_ip] = conversation_history[user_ip][-30:]

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "فيه مشكلة، جرب تاني"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
