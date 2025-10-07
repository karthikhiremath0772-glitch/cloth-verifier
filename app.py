import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
import os
from streamlit_qr_code_scanner import qr_code_scanner
from datetime import datetime
import csv

st.set_page_config(page_title="Cloth Return Verifier", layout="centered")
st.title("🧵 Cloth Return Verifier")

# Scan QR code to get product ID
st.subheader("📷 Scan Product QR Code")
qr_result = qr_code_scanner(key="qr")

if qr_result:
    product_id = qr_result
    st.success(f"✅ Scanned Product ID: {product_id}")
else:
    st.info("📌 Waiting for QR scan...")

# Upload returned cloth image
returned = st.file_uploader("Upload Returned Cloth Image", type=["jpg", "jpeg", "png"])

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load('models/resnet18_dummy.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Preprocess image
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Compare embeddings
def compare(img1, img2):
    emb1 = model(preprocess(img1)).squeeze().detach().numpy()
    emb2 = model(preprocess(img2)).squeeze().detach().numpy()
    score = cosine_similarity([emb1], [emb2])[0][0] * 100
    return score

# Run verification
if qr_result and returned:
    try:
        original_path = f'products/{product_id}/image.jpg'
        img1 = Image.open(original_path).convert('RGB')
        img2 = Image.open(returned).convert('RGB')

        st.image([img1, img2], caption=["Original", "Returned"], width=250)

        score = compare(img1, img2)
        st.write(f"🔍 Similarity Score: **{score:.2f}%**")

        if score > 90:
            decision = "✅ ORIGINAL"
            st.success(decision)
        elif score < 70:
            decision = "❌ FAKE"
            st.error(decision)
        else:
            decision = "⚠️ Manual Review"
            st.warning(decision)

        # Log result
        log_path = 'logs/verification_log.csv'
        os.makedirs('logs', exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Product ID', 'Similarity Score (%)', 'Decision'])

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), product_id, f"{score:.2f}", decision])
        st.info("📝 Verification result logged.")

    except FileNotFoundError:
        st.error("❌ Original image not found for this product ID.")
