import os
import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==== إعداد المسارات ====
image_dir = r"D:\EALPR-master (1)\images"
label_dir = r"D:\EALPR-master (1)\labels"
model_path = r"D:\EALPR-master (1)\working\runs\detect\yolo11m_car_plate\weights\best.pt"

# ==== تحميل الموديل ====
model = YOLO(model_path)

# ==== Mapping من class ID إلى حرف ====
CLASS_LABELS_MAPPING = {
    0: "٠", 1: "١", 2: "٢", 3: "٣", 4: "٤", 5: "٥", 6: "٦", 7: "٧",
    8: "ح", 9: "٨", 10: "٩", 11: "ط", 12: "ظ", 13: "ع", 14: "أ", 15: "ب",
    16: "ض", 17: "د", 18: "ف", 19: "غ", 20: "ه", 21: "ج", 22: "ك", 23: "خ",
    24: "ل", 25: "م", 26: "ن", 27: "ق", 28: "ر", 29: "ص", 30: "س", 31: "ش",
    32: "ت", 33: "ث", 34: "و", 35: "ي", 36: "ذ", 37: "ز"
}

y_true = []
y_pred = []

image_paths = glob(os.path.join(image_dir, "*.jpg"))

for img_path in image_paths:
    filename = os.path.basename(img_path).replace(".jpg", "")
    label_path = os.path.join(label_dir, f"{filename}.txt")

    if not os.path.exists(label_path):
        print(f"[SKIP] Missing label for {filename}")
        continue

    with open(label_path, "r") as f:
        line = f.readline()
        if not line:
            continue
        gt_class = int(line.strip().split()[0])
        y_true.append(gt_class)

    results = model(img_path, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy()

    if len(boxes) == 0:
        print(f"[SKIP] No prediction for {filename}")
        y_pred.append(-1)
        continue

    # ناخد أعلى ثقة فقط
    best_box = sorted(boxes, key=lambda x: x[4], reverse=True)[0]
    pred_class = int(best_box[5])
    y_pred.append(pred_class)

# ==== حذف الحالات اللي مفيهاش توقع ====
filtered_true = [t for t, p in zip(y_true, y_pred) if p != -1]
filtered_pred = [p for p in y_pred if p != -1]

# ==== حساب الدقة ====
acc = accuracy_score(filtered_true, filtered_pred)
print(f"\n✅ OCR Classification Accuracy: {acc * 100:.2f}%\n")

# ==== تقرير مفصل ====
print("📄 Classification Report:")
print(classification_report(filtered_true, filtered_pred, target_names=[
    CLASS_LABELS_MAPPING[c] for c in sorted(set(filtered_true + filtered_pred))
]))

# ==== Confusion Matrix ====
labels_sorted = sorted(set(filtered_true + filtered_pred))
label_names = [CLASS_LABELS_MAPPING[c] for c in labels_sorted]

cm = confusion_matrix(filtered_true, filtered_pred, labels=labels_sorted)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, xticklabels=label_names, yticklabels=label_names, annot=True, fmt="d", cmap="Blues")
plt.title("OCR Confusion Matrix")
plt.xlabel("Predicted Character")
plt.ylabel("True Character")
plt.tight_layout()
plt.savefig("ocr_confusion_matrix.png")
plt.show()
# ==== حفظ النتائج في ملف ====
results_file = "ocr_results.txt" 