import os
import re
import io
import base64
import requests
import pandas as pd
from PIL import Image
import google.generativeai as genai
from flask_cors import CORS
from dotenv import load_dotenv
from collections import defaultdict
from datetime import date, timedelta
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY مش موجود!")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel(
    'gemini-2.0-flash',
    generation_config={"temperature": 0.85, "max_output_tokens": 2048}
)

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
    except:
        return 25

# ==================== Memory ====================
conversation_history = defaultdict(list)
user_disabled_products = set()

# ==================== Gemini Chat ====================
def gemini_chat(user_message="", image_b64=None):
    try:
        user_ip = get_user_ip()
        location = get_location(user_ip)
        city = location["city"]
        today_temp = fetch_weather(location["lat"], location["lon"])

        products_text = "المنتجات المتاحة (لازم ترشح من دول بس لو اليوزر عايز لبس):\n"
        for _, row in CSV_DATA.iterrows():
            name = str(row.get('name') or row.get('اسم المنتج') or row.get('product_name') or row.iloc[0]).strip()
            price = row.get('price') or row.get('السعر') or row.iloc[1]
            cat = str(row.get('category') or row.get('الكاتيجوري') or row.get('القسم') or row.iloc[2] if len(row)>2 else "").strip()
            id_ = row.get('id') or row.get('product_id') or row.iloc[3] if len(row)>3 else "unknown"
            products_text += f"• {name} | السعر: {price} جنيه | الكاتيجوري: {cat} | ID: {id_}\n"

        clothing_keywords = ["لبس", "تيشرت", "بنطلون", "جاكيت", "قميص", "هودي", "كارديجان", "كوتشي", "ترينج", "جينز", "رشح", "عايز", "زي", "نفس", "شبه", "شبه", "اقترح"]

        user_text_lower = user_message.lower() if user_message else ""
        is_clothing_request = any(word in user_text_lower for word in clothing_keywords)

                full_message = f"""
أنت شاب مصري بتتكلم عامية مصرية طبيعية وودودة جدًا، بتعرف تحلل صور وبتفهم في الموضة كويس.

الجو في {city} النهاردة: {today_temp}°C

المنتجات اللي عندك (لازم تنسخ الاسم بالحرف من دول، من غير أي تغيير أو تلخيص أو اختراع):

{products_text}

≫≫ قواعد لا تُكسر أبدًا – لازم تتبعها بالحرف ≪≪

- لو اليوزر عايز لبس أو رفع صورة لبس → لازم ترشح منتجات من القايمة فوق بالتنسيق ده بالظبط، وتنسخ الاسم زي ما هو مكتوب في القايمة حتى لو الصورة مختلفة شوية.

ممنوع منعًا باتًا:
- تكتب "جاكيت جينز" أو "تيشيرت أبيض" أو أي اسم مختصر
- تختصر أو تعيد صياغة اسم المنتج
- تخترع اسم جديد

لازم تكتب الاسم زي ما هو في القايمة، مثال:

سكارف كشمير طويل
السعر: 290 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1014

كارديجان صوف خفيف بيج
السعر: 320 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1012

قميص فلانيل كاروهات أحمر وأسود
السعر: 380 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1011

≫≫ أمثلة إجبارية لازم تقلدها بالظبط ≪≪

لو اليوزر قال "عايز جاكيت":
سكارف كشمير طويل
السعر: 290 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1014

كارديجان صوف خفيف بيج
السعر: 320 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1012

لو رفع صورة جاكيت جينز وقال "عايز زي ده":
الله ينور الجاكيت ده شيك أوي، تعالى أوريك حاجات قريبة منه عندنا

سكارف كشمير طويل
السعر: 290 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1014

كارديجان صوف خفيف بيج
السعر: 320 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1012

≫≫ قواعد عامة ≪≪
- لو اليوزر مش بيسأل عن لبس أو رفع صورة عادية → حلل الصورة ورد طبيعي من غير منتجات
- لو رفع صورة + كلام عن لبس → حلل الصورة الأول وبعدين رشح بالتنسيق أعلاه
- لو رفع صورة بس → ابدأ بـ "ثانية بس أشوف الصورة..."
- ردك عامية مصرية طبيعية 100%
- متستخدمش إيموجي
- متعترفش إنك بوت

رد دلوقتي.
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
        return "ثواني بس وأرجعلك"

# ==================== Routes ====================
@app.route("/")
def home():
    return "شغال 100%"

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
            if image_file.content_length > 5*1024*1024:
                return jsonify({"error": "الصورة كبيرة"}), 400
            image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

        if not user_message and not image_b64:
            return jsonify({"error": "لازم تبعت حاجة"}), 400

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

