import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model locally (no GitHub calls)
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("models/resnet18_dummy.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# UI
st.title("🧠 Cloth Verifier App")
st.subheader("🔤 Enter Product ID")
product_id = st.text_input("Product ID", placeholder="e.g. shirt123")

uploaded_file = st.file_uploader("📤 Upload Return Image", type=["jpg", "jpeg", "png"])

if uploaded_file and product_id:
    # Load return image
    return_img = Image.open(uploaded_file).convert("RGB")
    return_tensor = transform(return_img).unsqueeze(0)

    # Load original image
    original_path = f"products/{product_id}/image.jpg"
    if os.path.exists(original_path):
        original_img = Image.open(original_path).convert("RGB")
        original_tensor = transform(original_img).unsqueeze(0)

        # Extract embeddings
        with torch.no_grad():
            return_embed = model(return_tensor).numpy()
            original_embed = model(original_tensor).numpy()

        # Compare
        similarity = cosine_similarity(return_embed, original_embed)[0][0]
        st.write(f"🧮 Similarity Score: `{similarity:.2f}`")

        if similarity > 0.8:
            st.success("✅ Return Verified: Product matches original.")
        else:
            st.error("❌ Return Rejected: Product does not match.")
    else:
        st.warning("⚠️ Original product image not found.")
elif product_id and not uploaded_file:
    st.info("📌 Please upload a return image to verify.")
elif uploaded_file and not product_id:
    st.info("📌 Please enter a product ID to proceed.")
