from ultralytics import YOLO

model = YOLO("best.pt")

# Method 1: Direct path
# results = model("test/")

# Method 2: Using predict() method
results = model.predict(r"test\Screenshot 2026-04-30 180730.png", imgsz=224)

# Get prediction details
result = results[0]
top_class = result.probs.top1
top_confidence = result.probs.top1conf
class_name = result.names[top_class]

print(f"Prediction: {class_name} (Confidence: {top_confidence:.2%})")