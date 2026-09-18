from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import uvicorn
from pathlib import Path

# ============================================================
# ============================================================
# ใช้ตำแหน่งของไฟล์ main.py เป็นโฟลเดอร์หลัก
# ทำให้หาไฟล์โมเดลและ labels ได้ถูกต้องบน Render และเครื่อง local
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_unquant.tflite"
LABELS_PATH = BASE_DIR / "labels.txt"
IMAGE_SIZE = (224, 224)

# โหลด TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# โหลด Labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

# สร้างแอป FastAPI
app = FastAPI(title="Avocado AI API", description="API สำหรับวิเคราะห์ภาพด้วย TFLite")

# ============================================================
# ฟังก์ชันประมวลผลหลัก (Core Logic)
# ============================================================
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """แปลงไบต์ภาพเป็น Tensor ที่พร้อมป้อนเข้าโมเดล"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # Normalize สำหรับ Teachable Machine
    return np.expand_dims(arr, axis=0)

def predict(image_bytes: bytes) -> list:
    """สั่งโมเดลทำนายผล"""
    data = preprocess_image(image_bytes).astype(input_details[0]["dtype"])
    
    interpreter.set_tensor(input_details[0]["index"], data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    
    results = [
        {"label": labels[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in range(len(labels))
    ]
    return sorted(results, key=lambda x: x["confidence"], reverse=True)

def get_recommendation(label: str, confidence: float) -> str:
    """สร้างคำแนะนำจากผลการจำแนกภาพ"""
    if confidence < 40:
        return "กรุณาถ่ายภาพอะโวคาโดใหม่ให้ชัดเจนขึ้น"

    recommendations = {
        "สายพันธุ์แฮส ดิบ": "อะโวคาโดยังดิบ ควรรอให้สุกอีก 2-4 วัน",
        "สายพันธุ์แฮส สุกพอดีทาน": "อะโวคาโดสุกพอดี พร้อมรับประทาน",
        "สายพันธุ์แฮส สุกเกินไป": "อะโวคาโดสุกเกินไป ควรรับประทานทันทีหรือนำไปทำอาหาร",
        "ไม่ใช่อะโวคาโด": "ไม่พบอะโวคาโด กรุณาถ่ายภาพใหม่"
    }
    return recommendations.get(label, "ไม่สามารถให้คำแนะนำสำหรับผลลัพธ์นี้ได้")

# ============================================================
# API Routes
# ============================================================
@app.get("/")
def home():
    return {"status": "ok", "message": "Avocado  API (FastAPI) กำลังทำงาน"}

@app.post("/predict")
async def predict_route(image: UploadFile = File(...)):

    print(f"DEBUG: ชื่อไฟล์={image.filename}, ประเภท={image.content_type}")
    # 1. ป้องกันคนส่งไฟล์มั่วที่ไม่ใช่รูปภาพ
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="กรุณาส่งไฟล์รูปภาพเท่านั้น")
    
    # 2. อ่านข้อมูลภาพแบบ Asynchronous
    image_bytes = await image.read()
    
    # 3. นำไปทำนายผล
    try:
        results = predict(image_bytes)
        top_result = results[0]
        return {
            "success": True,
            "top_result": top_result,
            "recommendation": get_recommendation(
                top_result["label"],
                top_result["confidence"]
            ),
            "all_results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ประมวลผลล้มเหลว: {str(e)}")

# ============================================================
# รันเซิร์ฟเวอร์ (สำหรับ Development)
# ============================================================
if __name__ == "__main__":
    # รันผ่าน uvicorn ตรงๆ ในโค้ด
    # Render กำหนดพอร์ตผ่านตัวแปร PORT ส่วนการรัน local ใช้พอร์ต 8000
    import os
    port = int(os.environ.get("PORT", 8000))

    # เปิดให้เข้าถึงจากภายนอก และปิด reload สำหรับ production
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
