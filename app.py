import streamlit as st

st.set_page_config(
    page_title="Tuberculosis Detection",
    page_icon="🫁",
    layout="wide"
)

st.sidebar.title("🫁 TB Detection")
st.sidebar.caption("Deep Learning • Chest X-ray Screening")

st.sidebar.divider()

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset",
        "🤖 Models",
        "🔍 Prediction",
        "ℹ️ About"
    ]
)

if page == "🏠 Home":

    st.title("🫁 Tuberculosis Detection Using Deep Learning")

    st.markdown(
        "### AI-powered Chest X-ray Screening"
    )

    st.write(
        "A deep learning-based system that classifies chest X-ray images "
        "as **Normal** or **Tuberculosis (TB)** using transfer learning."
    )

    st.divider()

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset", "3,008 X-rays")

    with col2:
        st.metric("Models", "3")

    with col3:
        st.metric("TB Sensitivity", "98.13%")

    with col4:
        st.metric("ROC-AUC", "0.9998")

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        "The objective is to build a reliable chest X-ray screening system "
        "that can identify images showing signs of Tuberculosis while also "
        "correctly recognizing Normal X-rays."
    )

    st.subheader("⚙️ Project Workflow")

    workflow_col1, workflow_col2, workflow_col3, workflow_col4 = st.columns(4)

    with workflow_col1:
        st.info("**1. Dataset**\n\nChest X-ray collection")

    with workflow_col2:
        st.info("**2. Preprocessing**\n\nResize and prepare images")

    with workflow_col3:
        st.info("**3. Deep Learning**\n\nTrain transfer-learning models")

    with workflow_col4:
        st.info("**4. Evaluation**\n\nCompare model performance")

    st.divider()

    st.subheader("🤖 Models Evaluated")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.write("### ResNet50")
        st.success("Final selected model")

    with model_col2:
        st.write("### VGG16")
        st.info("Evaluated Model")

    with model_col3:
        st.write("### EfficientNetB0")
        st.info("Evaluated Model")

    st.divider()

    st.subheader("🏆 Final Model")

    st.success("ResNet50")

    st.write(
        "ResNet50 achieved **98.13% TB sensitivity**, **100% Normal "
        "specificity**, and **0.9998 ROC-AUC** on the test dataset."
    )

    st.warning(
        "⚠️ This application is intended for screening and educational "
        "purposes only. It is not a substitute for professional medical diagnosis."
    )

elif page == "📊 Dataset":

    st.title("📊 Dataset Explorer")

    st.write(
        "The dataset contains chest X-ray images divided into two classes: "
        "Normal and Tuberculosis (TB)."
    )

    st.divider()

    # Dataset Overview
    st.subheader("📌 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", "3,008")

    with col2:
        st.metric("Normal", "514")

    with col3:
        st.metric("TB", "2,494")

    with col4:
        st.metric("Image Classes", "2")

    st.divider()

    # Class Distribution
    st.subheader("📈 Class Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Normal X-rays**")
        st.metric("514 images", "17.1% of dataset")
        st.progress(514 / 3008)

    with col2:
        st.write("**TB X-rays**")
        st.metric("2,494 images", "82.9% of dataset")
        st.progress(2494 / 3008)

    st.divider()

    # Dataset Split
    st.subheader("🔀 Dataset Split")

    split_col1, split_col2, split_col3 = st.columns(3)

    with split_col1:
        st.metric("Training", "2,104", "70%")

    with split_col2:
        st.metric("Validation", "451", "15%")

    with split_col3:
        st.metric("Testing", "453", "15%")

    st.write("### Class-wise Split")

    split_data = {
        "Dataset": ["Training", "Validation", "Testing"],
        "Normal": [359, 77, 78],
        "TB": [1745, 374, 375],
        "Total": [2104, 451, 453]
    }

    st.dataframe(
        split_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Preprocessing
    st.subheader("⚙️ Image Preprocessing")

    preprocessing_col1, preprocessing_col2 = st.columns(2)

    with preprocessing_col1:
        st.write("**Image Size**")
        st.write("224 × 224 pixels")

        st.write("**Color Format**")
        st.write("RGB")

    with preprocessing_col2:
        st.write("**Dataset Split**")
        st.write("70% Training / 15% Validation / 15% Testing")

        st.write("**Model Input**")
        st.write("Preprocessed according to the selected pretrained architecture")

    st.divider()

    st.subheader("🔎 Dataset Structure")

    st.code(
        """Dataset
├── Normal Chest X-rays
│   └── 514 images
│
└── TB Chest X-rays
    └── 2,494 images""",
        language="text"
    )

elif page == "🤖 Models":

    st.title("🤖 Model Comparison")

    st.write(
        "Three transfer learning models were trained and evaluated for "
        "Tuberculosis detection."
    )

    st.subheader("Models Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### ResNet50\nPrimary final model")

    with col2:
        st.info("### VGG16\nTransfer learning model")

    with col3:
        st.info("### EfficientNetB0\nTransfer learning model")

    st.subheader("Performance Comparison")

    comparison_data = {
        "Model": [
            "ResNet50",
            "VGG16",
            "EfficientNetB0"
        ],
        "TB Sensitivity": [
            "98.13%",
            "92.27%",
            "93.87%"
        ],
        "Normal Specificity": [
            "100.00%",
            "100.00%",
            "97.44%"
        ],
        "ROC-AUC": [
            "0.9998",
            "0.9893",
            "0.9918"
        ],
        "False Negatives": [
            7,
            29,
            23
        ],
        "False Positives": [
            0,
            0,
            2
        ]
    }

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Evaluation Criteria")

    st.write(
        "**Primary Metric:** TB Sensitivity (Recall)  \n"
        "**Secondary Metrics:** Normal Specificity and ROC-AUC"
    )

    st.subheader("Selected Model")

    st.success("ResNet50")

    st.write(
        "ResNet50 was selected as the final model based on the project's "
        "evaluation criteria, with 98.13% TB sensitivity, 100% Normal "
        "specificity, and 0.9998 ROC-AUC on the test set."
    )

    st.info(
        "False negatives represent TB-positive X-rays that were predicted "
        "as Normal. These errors are specifically monitored during evaluation."
    )

elif page == "🔍 Prediction":

    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from PIL import Image

    st.title("🔍 TB Prediction")

    st.write(
        "Upload a chest X-ray image to classify it as **Normal** or "
        "**Tuberculosis (TB)**."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Unable to read the uploaded image.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Chest X-ray",
                width="stretch"
            )

        with col2:
            st.subheader("Image Information")

            st.write(f"**Format:** {image.format or 'Image'}")
            st.write(f"**Original Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write("**Model Input:** 224 × 224 pixels")
            st.write("**Model:** ResNet50")

        st.divider()

        image_array = np.array(image)

        image_array = tf.image.resize(
            image_array,
            (224, 224)
        )

        image_array = preprocess_input(image_array)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        with st.spinner("🔬 Analyzing X-ray..."):

            model = tf.keras.models.load_model(
                "final_tb_model.keras"
            )

            prediction = model.predict(
                image_array,
                verbose=0
            )[0][0]

        if prediction >= 0.5:
            result = "TB"
            confidence = prediction
        else:
            result = "Normal"
            confidence = 1 - prediction

        st.subheader("Prediction Result")

        if result == "TB":

            st.error(
                f"### Result: {result}"
            )

            st.write(
                f"Model confidence: **{confidence * 100:.2f}%**"
            )

        else:

            st.success(
                f"### Result: {result}"
            )

            st.write(
                f"Model confidence: **{confidence * 100:.2f}%**"
            )

        st.divider()

        st.warning(
            "⚠️ This result is intended for screening and educational purposes "
            "only. It is not a medical diagnosis and should be reviewed by a "
            "qualified healthcare professional."
        ) 

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.write(
        "Tuberculosis Detection Using Deep Learning is a computer vision "
        "project designed to classify chest X-ray images as Normal or "
        "Tuberculosis (TB)."
    )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        "The objective is to develop a deep learning-based screening system "
        "that can identify TB-related patterns in chest X-ray images while "
        "also correctly recognizing Normal X-rays."
    )

    st.subheader("🧠 Technical Approach")

    st.write(
        "Transfer learning was used with three pretrained CNN architectures: "
        "ResNet50, VGG16, and EfficientNetB0. Their performance was evaluated "
        "on a separate test dataset."
    )

    st.divider()

    st.subheader("🛠️ Technologies")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.write("**Programming & Frameworks**")
        st.write("• Python")
        st.write("• TensorFlow")
        st.write("• Keras")
        st.write("• Streamlit")

    with tech_col2:
        st.write("**Data & Computer Vision**")
        st.write("• OpenCV")
        st.write("• Pillow")
        st.write("• NumPy")
        st.write("• Scikit-learn")

    st.divider()

    st.subheader("🤖 Deep Learning Models")

    st.write(
        "• ResNet50  \n"
        "• VGG16  \n"
        "• EfficientNetB0"
    )

    st.subheader("📈 Final Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("TB Sensitivity", "98.13%")

    with col2:
        st.metric("Normal Specificity", "100%")

    with col3:
        st.metric("ROC-AUC", "0.9998")

    st.divider()

    st.subheader("👨‍💻 Author")

    st.write("**Sachin Kumar Rao**")

    st.write("Data Science & AI/ML Enthusiast")

    st.write(
        "GitHub: [github.com/rao274563-cpu]"
        "(https://github.com/rao274563-cpu)"
    )

    st.write(
        "LinkedIn: [linkedin.com/in/sachin-rao-535b0b331]"
        "(https://www.linkedin.com/in/sachin-rao-535b0b331)"
    )

    st.divider()

    st.caption(
        "⚠️ This application is intended for educational and screening "
        "purposes only and should not be used as a substitute for professional "
        "medical diagnosis."
    )