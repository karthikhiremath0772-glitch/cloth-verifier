import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import qrcode
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained ResNet18 model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # Use embeddings directly
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🧠 Cloth Verifier App")

# -------------------- Customer Side --------------------
st.header("👤 Customer: Upload Product & Generate QR")

product_id = st.text_input("Enter Product ID", placeholder="e.g. shirt123")
product_img = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"], key="product")

if product_id.strip() and product_img:
    product_folder = f"products/{product_id}"
    os.makedirs(product_folder, exist_ok=True)

    product_path = f"{product_folder}/image.jpg"
    Image.open(product_img).save(product_path)
    st.success(f"✅ Product image saved for ID: {product_id}")
    st.image(product_path, caption="🧥 Uploaded Product Image")

    # Generate QR
    qr = qrcode.make(product_id)
    qr_path = f"{product_folder}/qr.png"
    qr.save(qr_path)
    st.image(qr_path, caption="📦 QR Code for Packaging")

# -------------------- Delivery Side --------------------
st.header("🚚 Delivery: Upload Return & Verify")

return_img = st.file_uploader("Upload Return Image", type=["jpg", "jpeg", "png"], key="return")
verify_id = st.text_input("Scan or Enter QR Code (Product ID)", placeholder="e.g. shirt123", key="verify")

if return_img and verify_id.strip():
    os.makedirs("returns", exist_ok=True)
    return_path = f"returns/{verify_id}_return.jpg"
    Image.open(return_img).save(return_path)
    st.success(f"📤 Return image saved for ID: {verify_id}")
    st.image(return_path, caption="📸 Uploaded Return Image")

    # Compare
    original_path = f"products/{verify_id}/image.jpg"
    if os.path.exists(original_path):
        def get_embedding(path):
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                return model(tensor).numpy()

        original = get_embedding(original_path)
        returned = get_embedding(return_path)
        similarity = cosine_similarity(original, returned)[0][0]

        col1, col2 = st.columns(2)
        col1.image(original_path, caption="🧥 Original Product Image")
        col2.image(return_path, caption="📸 Return Image")

        st.write(f"🧮 Similarity Score: `{similarity:.2f}`")
        if similarity > 0.8:
            st.success("✅ Return Verified: Product matches original.")
        else:
            st.error("❌ Return Rejected: Product does not match.")
    else:
        st.warning("⚠️ Original product image not found.")
