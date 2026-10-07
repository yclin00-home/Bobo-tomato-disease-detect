import json
import os
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

# =============================================================
# 1. 載入模型權重與類別標籤
# =============================================================
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

if not os.path.exists("class_names.json"):
    raise FileNotFoundError("❌ 找不到 class_names.json！")

with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

num_classes = len(class_names)

if not os.path.exists("tomato_resnet18.pth"):
    raise FileNotFoundError("❌ 找不到 tomato_resnet18.pth！")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("tomato_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# 影像轉換規則 (與訓練時一致)
val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# =============================================================
# 2. 病斑面積比例計算 (保持與第一段原解析度計算邏輯一致)
# =============================================================
def calculate_infection_ratio_from_cv(img_bgr):
    # 轉換顏色空間
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. 葉片範圍判定
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    lower_diseased = np.array([10, 30, 30])
    upper_diseased = np.array([35, 255, 255])
    mask_diseased = cv2.inRange(img_hsv, lower_diseased, upper_diseased)

    leaf_mask = cv2.bitwise_or(mask_green, mask_diseased)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    # 2. 病斑範圍判定
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    mask_dark = cv2.inRange(img_hsv, lower_dark, upper_dark)

    raw_lesion_mask = cv2.bitwise_or(mask_diseased, mask_dark)
    lesion_mask = cv2.bitwise_and(
        raw_lesion_mask, raw_lesion_mask, mask=leaf_mask
    )
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

    # 3. 面積統計
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    lesion_pixels = cv2.countNonZero(lesion_mask)

    if total_leaf_pixels == 0:
        return 0.0, leaf_mask, lesion_mask

    infection_ratio = (lesion_pixels / total_leaf_pixels) * 100.0

    # 轉為 3 通道以確保 Gradio 順暢顯示
    leaf_mask_rgb = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2RGB)
    lesion_mask_rgb = cv2.cvtColor(lesion_mask, cv2.COLOR_GRAY2RGB)

    return infection_ratio, leaf_mask_rgb, lesion_mask_rgb


# =============================================================
# 3. Gradio 網頁介面專用診斷邏輯
# =============================================================
def predict_tomato_disease(input_pil_img):
    if input_pil_img is None:
        return None, "No image selected", "Please upload or click to select an image.。", None, None

    # 自動修正 EXIF 圖片轉向
    input_pil_img = ImageOps.exif_transpose(input_pil_img)

    # 1. AI 深度學習模型預測
    input_tensor = val_transforms(input_pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    confidences = {
        class_names[i]: float(probs[i]) for i in range(num_classes)
    }

    top_pred = max(confidences, key=confidences.get)
    pred_label_lower = top_pred.lower()

    # 2. 判斷是否為晚疫病並進行病斑計算 (對齊第一段邏輯)
    if "late" in pred_label_lower or "blight" in pred_label_lower:
        img_np = np.array(input_pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        infection_ratio, leaf_mask_rgb, lesion_mask_rgb = (
            calculate_infection_ratio_from_cv(img_bgr)
        )

        ratio_str = f"{infection_ratio:.2f}%"

        if infection_ratio < 15.0:
            advice = f"💡 **Protection recommendation**：Mild late blight infection (Lesions account for {infection_ratio:.1f}%)，Please immediately remove the diseased leaves and apply an organocopper compound for protection.。"
        elif infection_ratio < 45.0:
            advice = f"⚠️ **Protection recommendation**：Moderate late blight infection (Lesions account for {infection_ratio:.1f}%)，The plant needs to be isolated and a systemic fungicide used to control its spread."
        else:
            advice = f"🚨 **Warning**：Severe late blight infection (Lesions account for {infection_ratio:.1f}%)！It is recommended to remove and destroy the plant to prevent the spread of spores."

        return (
            confidences,
            ratio_str,
            advice,
            leaf_mask_rgb,
            lesion_mask_rgb,
        )
    else:
        # 非晚疫病或健康狀態
        advice = f"✅ **Test Result:** Predicted as **{top_pred}**. Leaves are in good condition or not late blight; no need to calculate lesion area."
        return confidences, "N/A", advice, None, None


# =============================================================
# 4. 建立 Gradio UI
# =============================================================
example_images = []
if os.path.exists("./tomato-test.jpg"):
    example_images.append(["./tomato-test.jpg"])

with gr.Blocks(title="AI Diagnostic System for Tomato Leaf Diseases") as demo:
    gr.Markdown("#AI-based diagnosis and lesion area analysis system for tomato leaf diseases")
    gr.Markdown(
        "You can **click to upload and select a picture from your computer**, **paste a picture from your scrapbook**, or **click the example image below** for diagnosis:"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Select or upload an image",
                sources=["upload", "clipboard"],
            )

            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label="💡 Click on the example image below to test directly:",
                )

            btn_submit = gr.Button("🔍 Start diagnosis", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="AI Category Recognition and Confidence Level (Confidence Level)"
            )
            ratio_output = gr.Textbox(
                label="Lesion area ratio (Infection Ratio)"
            )
            advice_output = gr.Markdown(label="Diagnostic and protective recommendations")

    with gr.Row():
        leaf_mask_output = gr.Image(
            label="Leaf range masking (Leaf Mask)", type="numpy"
        )
        lesion_mask_output = gr.Image(
            label="Covering the lesion area (Lesion Mask)", type="numpy"
        )

    btn_submit.click(
        fn=predict_tomato_disease,
        inputs=[image_input],
        outputs=[
            label_output,
            ratio_output,
            advice_output,
            leaf_mask_output,
            lesion_mask_output,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
