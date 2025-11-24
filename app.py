import os
import re
import base64
import requests
import pandas as pd
from flask_cors import CORS
from dotenv import load_dotenv
from collections import defaultdict
from datetime import date, timedelta
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==================== xAI Grok Config ====================
GROK_API_KEY = os.environ.get("GROK_API_KEY")
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY مش موجود في الـ environment variables!")

from xai_sdk import Client

client = Client(api_key=GROK_API_KEY)

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
        except:
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
    except:
        return None

def suggest_outfit(temp, rain):
    if rain > 2.0: return "مطر – خُد شمسية"
    if temp < 10: return "برد جدًا – جاكيت تقيل"
    if temp < 18: return "بارد – جاكيت خفيف"
    if temp < 26: return "معتدل – تيشيرت وجينز"
    if temp < 32: return "دافئ – تيشيرت خفيف"
    return "حر – شورت ومياه كتير"

conversation_history = defaultdict(list)

def grok_chat(messages):
    try:
        chat = client.chat.create(model="grok-4")
        for msg in messages:
            if msg["role"] == "system":
                chat.append(system(msg["content"]))
            elif msg["role"] == "user":
                if isinstance(msg["content"], list):
                    text_content = next((item["text"] for item in msg["content"] if item["type"] == "text"), "")
                    image_b64_url = next((item["image_url"]["url"] for item in msg["content"] if item["type"] == "image_url"), None)
                    if image_b64_url:
                        chat.append(user(text_content, image(image_b64_url)))
                    else:
                        chat.append(user(text_content))
                else:
                    chat.append(user(msg["content"]))
            elif msg["role"] == "assistant":
                chat.append(assistant(msg["content"]))
        
        response = chat.sample(temperature=0.7, max_tokens=1024)
        return response.content.strip()
    except Exception as e:
        return "المودل مش شغال حاليا!"

@app.route("/")
def home():
    return jsonify({
        "message": "PureSoft AI Backend شغال 100%",
        "api": "/api/chat",
        "frontend": "https://mohamedahmed517.github.io/PureSoft_Website/"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        user_ip = get_user_ip()
        location = get_location(user_ip)
        if not location:
            return jsonify({"error": "مش عارف أحدد مكانك"}), 400

        city = location["city"]
        weather_data = fetch_weather(location["lat"], location["lon"])
        if not weather_data:
            return jsonify({"error": "مشكلة في جلب الطقس"}), 500

        today = date.today()
        forecast_lines = []
        for i in range(min(14, len(weather_data["time"]))):
            d = (today + timedelta(days=i)).strftime("%d-%m")
            t_max = weather_data["temperature_2m_max"][i]
            t_min = weather_data["temperature_2m_min"][i]
            temp = round((t_max + t_min) / 2, 1)
            rain = weather_data["precipitation_sum"][i]
            outfit = suggest_outfit(temp, rain)
            forecast_lines.append(f"{d}: {temp}°C – {outfit}")

        forecast_text = "\n".join(forecast_lines)

        user_message = request.form.get("message", "").strip()
        image_file = request.files.get("image")
        image_b64 = None
        if image_file:
            if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                return jsonify({"error": "نوع الصورة مش مدعوم"}), 400
            if image_file.content_length > 5 * 1024 * 1024:
                return jsonify({"error": "الصورة كبيرة أوي (حد أقصى 5 ميجا)"}), 400
            image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

        if not user_message and not image_b64:
            return jsonify({"error": "لازم تبعت رسالة أو صورة"}), 400

        history = conversation_history[user_ip]

        messages = [
                    {"role": "system", "content": f"""
        أنت مساعد مصري ذكي وودود جدًا، بتتكلم عامية مصرية طبيعية 100%، ممتع وصريح وبتفهم اليوزر من نص كلمة.

        المنتجات اللي عندنا (لازم ترشح منهم فقط لو طلب أي حاجة للبيع):
        {CSV_DATA}

        توقعات الطقس في {city} لمدة 14 يوم:
        {forecast_text}

        ★★★★★ قواعد صارمة جدًا ★★★★★
        1. أول رد بس: رحب بسيط + قول المدينة والطقس مرة واحدة بشكل طبيعي.
        2. بعدها ما تعيدش الطقس أبدًا إلا لو سأل صراحة.
        3. لو رفع صورة → حللها كويس (لبس، بشرة، شعر، مكان، جو...) ورشح منتجات مناسبة.
        4. ★★★ أهم حاجة ★★★
        كل ما ترشح أي منتج (واحد أو أكتر)، لازم تكتب كل منتج بالشكل ده بالظبط وما ينفعش تغير الترتيب أبدًا:

        🛍️ **اسم المنتج**
        💰 السعر: xxx جنيه
        📂 الكاتيجوري: كذا
        🔗 اللينك: https://afaq-stores.com/product-details/{{id}}

        مثال:
        🛍️ **جاكيت جلد اسود تقيل مبطن فرو**
        💰 السعر: 720 جنيه
        📂 الكاتيجوري: لبس شتوي
        🔗 اللينك: https://afaq-stores.com/product-details/1001

        5. لو فيه عرض أو خصم → اذكره بوضوح جنب السعر.
        6. لو اليوزر عايز يقعد في البيت → اقترح نشاطات بيتية وما تحفزش على الخروج أبدًا.
        7. ردودك دايمًا طبيعية جدًا زي البني آدمين، ممتعة، ومفيدة.
        8. اتكلم مصري 100%، مفيش فصحى ولا ألقاب زي "يا باشا" أو "يا فندم".
        """}]

        for role, text in history:
            messages.append({"role": role, "content": text})

        user_content = [{"type": "text", "text": user_message or "فيه صورة مرفوعة"}]
        if image_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })

        messages.append({"role": "user", "content": user_content})

        reply = grok_chat(messages)

        conversation_history[user_ip].append(("user", user_message or "[صورة]"))
        conversation_history[user_ip].append(("assistant", reply))
        if len(conversation_history[user_ip]) > 10:
            conversation_history[user_ip] = conversation_history[user_ip][-10:]

        return jsonify({
            "reply": reply,
            "city": city,
            "type": "chat",
            "has_image": bool(image_b64)
        })

    except Exception as e:
        print(f"خطأ عام: {e}")
        return jsonify({"error": "فيه مشكلة، حاول تاني"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
