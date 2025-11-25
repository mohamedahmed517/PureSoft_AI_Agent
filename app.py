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
from datetime import date, timedelta
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==================== Gemini Setup ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY مش موجود!")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel(
    'gemini-2.0-flash',
    generation_config={"temperature": 0.85, "max_output_tokens": 2048}
)

# ==================== CSV Data ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'products.csv')
CSV_DATA = pd.read_csv(csv_path)

# ==================== IP & Weather (اختياري) ====================
def get_location(ip):
    try:
        r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=8)
        r.raise_for_status()
        d = r.json()
        if d.get("city"):
            return d.get("city")
    except:
        pass
    return "القاهرة"

def fetch_weather(city="Cairo"):
    try:
        r = requests.get(f"https://wttr.in/{city}?format=%t", timeout=10)
        temp = r.text.strip().replace("+", "").replace("C", "")
        return float(temp) if temp.replace(".", "").isdigit() else 25
    except:
        return 25

# ==================== Memory ====================
conversation_history = defaultdict(list)

# ==================== Gemini Chat ====================
def gemini_chat(user_message="", image_b64=None):
    try:
        city = get_location(request.remote_addr or "127.0.0.1")
        today_temp = fetch_weather(city)

        products_text = "المنتجات المتاحة (لازم ترشح من دول بس لو اليوزر بيسأل عن لبس):\n"
        for _, row in CSV_DATA.iterrows():
            name = str(row.get('name') or row.get('اسم المنتج') or row.get('product_name') or row.iloc[0]).strip()
            price = row.get('price') or row.get('السعر') or row.iloc[1]
            cat = str(row.get('category') or row.get('الكاتيجوري') or row.get('القسم') or row.iloc[2] if len(row) > 2 else "").strip()
            id_ = row.get('id') or row.get('product_id') or row.iloc[3] if len(row) > 3 else "unknown"
            products_text += f"• {name} | السعر: {price} جنيه | الكاتيجوري: {cat} | ID: {id_}\n"

        user_text_lower = user_message.lower() if user_message else ""
        is_clothing_related = any(word in user_text_lower for word in ["لبس", "تيشرت", "بنطلون", "جاكيت", "قميص", "هودي", "كارديجان", "كوتشي", "ترينج", "جينز", "رشح", "عايز", "نفس", "زي"])

        full_message = f"""
أنت شاب مصري بتتكلم عامية مصرية طبيعية وودودة جدًا، بتعرف تحلل صور وتتكلم عن لبس كويس.

الجو النهاردة في {city}: {today_temp}°C

المنتجات اللي ممكن ترشحها (لازم تنسخ الاسم بالظبط من غير تغيير):
{products_text}

المحادثة السابقة:
{chr(10).join([t for _, t in conversation_history[request.remote_addr or "unknown"][-8:]])}

اليوزر{( " بيقول: " + user_message) if user_message else " مرفوع صورة بس"}

لو اليوزر بيسأل عن لبس أو عايز اقتراح (زي "رشحلي" أو "عايز جاكيت" أو "نفس ده") → لازم ترشح منتجات من القايمة فوق بنفس التنسيق ده بالظبط:

كارديجان صوف خفيف بيج
السعر: 320 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1012

قميص فلانيل كاروهات أحمر وأسود
السعر: 380 جنيه
الكاتيجوري: لبس خريفي
اللينك: https://afaq-stores.com/product-details/1011

ممنوع منعًا باتًا تغير اسم المنتج أو تكتب "لبس خريفي" كاسم أو تختصر.

لو اليوزر مرفوع صورة بس أو بيسأل حاجة عادية (مش عن لبس) → حلل الصورة ورد رد طبيعي من غير ما تجيب منتجات خالص.

لو اليوزر بيسأل عن لبس + مرفوع صورة → حلل الصورة وبعدين رشح منتجات بنفس التنسيق.

- ردك عامية مصرية طبيعية 100%
- متستخدمش إيموجي نهائي
- متكتبش إنك بوت أبدًا
- لو رفع صورة → ابدأ بـ "ثانية بس أشوف الصورة..."

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
        user_ip = request.remote_addr or "unknown"
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
