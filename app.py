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

# ==================== Gemini API ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY مش موجود! روح https://aistudio.google.com/app/apikey وخد واحد ببلاش")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel(
    'gemini-1.5-flash-exp-03-25',
    generation_config={"temperature": 0.7, "max_output_tokens": 1024}
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

def suggest_outfit(temp, rain):
    if rain > 2.0: return "مطر – خُد شمسية"
    if temp < 10: return "برد جدًا – جاكيت تقيل"
    if temp < 18: return "بارد – جاكيت خفيف"
    if temp < 26: return "معتدل – تيشيرت وجينز"
    if temp < 32: return "دافئ – تيشيرت خفيف"
    return "حر – شورت ومياه كتير"

conversation_history = defaultdict(list)

def gemini_chat(system_prompt, user_message, image_b64=None):
    try:
        user_ip = get_user_ip()
        chat = MODEL.start_chat()

        if len(conversation_history[user_ip]) == 0:
            chat.send_message(system_prompt)

        for role, text in conversation_history[user_ip]:
            if role == "user":
                chat.send_message(text)
            else:
                chat.send_message(text, role="model")

        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes))
            user_input = user_message or "حلل الصورة دي كويس جدًا ووصفها بالتفصيل"
            response = chat.send_message([user_input, img])
        else:
            user_input = user_message or "هاي"
            response = chat.send_message(user_input)

        return response.text.strip()

    except Exception as e:
        print(f"Gemini Error: {e}")
        return "ثواني بس وهرجعلك تاني!"

@app.route("/")
def home():
    return jsonify({
        "message": "PureSoft AI شغال 100% مع Gemini 1.5 Flash",
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

        system_prompt = f"""
        أنت مساعد مصري ذكي جدًا وودود، بتتكلم عامية مصرية سليمة وطبيعية 100% بدون أي خطأ إملائي أو نحوي.
        تفهم من نص كلمة وبترد بطريقة ممتعة ومباشرة وذكية.

        معلومات مهمة عندك دلوقتي:
        - اليوزر في مدينة: {city}
        - توقعات الطقس لمدة 14 يوم:
        {forecast_text}

        المنتجات اللي عندنا (ترشح منهم فقط لو طلب حاجة للبيع):
        {CSV_DATA.to_string(index=False)}

        ★★★★★ قواعد صارمة جدًا لازم تتبعها ★★★★★
        1. أول رد بس: رحب بسيط طبيعي + اذكر المدينة والجو النهاردة مرة واحدة فقط (مثلاً: "النهاردة في {city} الجو معتدل حلو أوي").
        2. بعدها متكررش الطقس أبدًا إلا لو سألك صراحة.
        3. متسألش أبدًا "عنوانك فين؟" أو "بتسألني كتير ليه؟" أو "بتعمل إيه؟" من غير سبب واضح.
        4. لو سألك "عامل إيه؟" → رد: "كويس الحمد لله، وانت؟"
        5. لو رفع صورة → حللها بدقة جدًا (شكل اللبس، لون البشرة، الشعر، المكان، الجو، المود...) ورشح منتجات مناسبة من اللي عندنا فقط.
        6. اتكلم عامية مصرية صح 100%، بدون ألقاب زي "يا باشا"، "يا وحش"، "يا ريس"، "يا معلم"، "يا برو"، "يا كبير"، إلخ.
        7. لما ترشح منتجات → ارشح كل منتج مرة واحدة فقط بالشكل ده بالظبط:
           تيشيرت كم طويل قطن ابيض
           السعر: 130 جنيه
           الكاتيجوري: لبس ربيعي
           اللينك: https://afaq-stores.com/product-details/1015
           → مفيش تكرار أبدًا في نفس الرد. لو طلب غيرهم → جيبله غيرهم.
        8. احفظ كل حاجة اليوزر يقولها عن نفسه (مثل: بحب الأسود، أنا مهندس، بشرتي فاتحة، إلخ) وخد بالك منها في كل ردودك القادمة في نفس الجلسة.
        9. لو اليوزر عمل ريفريش للصفحة → الذاكرة هتتمسح تلقائي وتبدأ من جديد (عادي جدًا).
        10. ردودك دايمًا طبيعية، ذكية، ومباشرة، وممتعة.

        ابدأ دلوقتي وخليك طبيعي جدًا.
        """

        reply = gemini_chat(system_prompt, user_message, image_b64)

        conversation_history[user_ip].append(("user", user_message or "[صورة]"))
        conversation_history[user_ip].append(("assistant", reply))
        if len(conversation_history[user_ip]) > 20:
            conversation_history[user_ip] = conversation_history[user_ip][-20:]

        return jsonify({
            "reply": str(reply),
            "city": city,
            "type": "chat",
            "has_image": bool(image_b64)
        })

    except Exception as e:
        print(f"خطأ عام: {e}")
        return jsonify({"error": "فيه مشكلة، حاول تاني بعد شوية"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
