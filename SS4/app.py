from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.saving import load_model
from tensorflow.keras.preprocessing import image

labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del'
]

def preprocess_PIL(pil_img, input_size=(128, 128)):
    pil_img = pil_img.convert("RGB")                                # Convert về RGB (tránh lỗi với hình RGBA)
    img = pil_img.resize(input_size)                                # Resize hình cho phù hợp với input size của mô hình
    img_array = image.img_to_array(img)                             # Chuyển từ kiểu dữ liệu hình sang kiểu numpy array

    img_array = np.expand_dims(img_array, axis=0)                   # Thêm n=1 để batch_size=1
    test_datagen = image.ImageDataGenerator(                        # Bắt buộc áp dụng các phương pháp tiền xử lý như tập train
        samplewise_center=True,            
        samplewise_std_normalization=True
    )
    img_generator = test_datagen.flow(img_array, batch_size=1)      # Thay vì sử dụng `flow_from_directory` thì chỉ sử dụng `flow`
    return img_generator

def main():
    @st.cache_resource
    def load_asl_model(model_path='model.keras'):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_asl_model('model_epoch_04.keras')
    st.title("American Sign Language (ASL) Classification App")

    option = st.selectbox("Choose input type", ("Upload Image", "Use Webcam"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True) # PIL object
            if st.button("Classify"):
                img_gen = preprocess_PIL(image)
                predictions = model.predict(next(img_gen))
                prediction_idx = np.argmax(predictions)
                predicted_label = labels[prediction_idx]
                confidence = np.max(predictions)
                st.write(f"**Prediction:** {predicted_label} with {confidence*100:.2f}% confidence.")
                
    elif option == "Use Webcam":
        st.write("Real-time ASL classification using your webcam.")
        st.write("TBD...")

if __name__ == "__main__":
    main()
