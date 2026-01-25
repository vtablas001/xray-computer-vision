# AI-Powered Chest X-Ray Diagnosis for El Salvador's Ministry of Health

This project leverages **computer vision and artificial intelligence (AI)** to support **early, automated detection of respiratory diseases** through chest X-Ray analysis. Developed in partnership with the **Ministry of Health (MINSAL)** and aligned with **UNDP’s 2025 AI for Sustainable Development Atlas**, the system aims to **empower medical professionals with AI tools** for faster and more accurate diagnosis, especially in highly demanded and resource-constrained settings.

---

## Project Context

Respiratory diseases remain a major public health challenge in El Salvador. This initiative is designed to help:

- **Reduce diagnostic delays**
- **Alleviate the burden on radiologists**
- **Improve healthcare access in underserved areas**

Inspired by global best practices, similar AI models have shown diagnostic performance comparable to **six independent radiologists**. This offers a promising pathway for supporting national health systems.

---

## Target Conditions

The AI system is trained to detect and classify the following diseases based on chest X-ray images:

1. **Pulmonary Tuberculosis**
2. **Pneumonia** (bacterial or viral)
3. **COVID-19**

These conditions were selected for their **high prevalence**, **radiographic visibility**, and **impact on the Salvadoran healthcare system**.

---

## Technical Approach

- **Transfer Learning**: We employ a pre-trained **VGG16** Convolutional Neural Network (CNN) from ImageNet, fine-tuned for medical image classification.
- **Data Augmentation**: Techniques such as **rotation**, **cropping**, and **brightness adjustment** are applied to enhance model generalization without requiring massive datasets.
- **Model Deployment**: The AI model is built using **open-source tools** in Python to ensure **sustainability, transparency, and local capacity building**.

---

## Language & Core Tools

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Data%20Validation-0E7C86?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI%20Server-121212?logo=uvicorn&logoColor=white)](https://www.uvicorn.org/)


## Libraries

[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-orange?logo=matplotlib)](https://matplotlib.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![OS](https://img.shields.io/badge/os-Standard%20Library-lightgrey)](https://docs.python.org/3/library/os.html)
[![Random](https://img.shields.io/badge/random-Standard%20Library-lightgrey)](https://docs.python.org/3/library/random.html)

## Demo
https://v0-pulmonary-ai-demo.vercel.app/
