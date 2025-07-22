# 🧠 AI-Powered Chest X-Ray Diagnosis for El Salvador's Ministry of Health

This project leverages **computer vision and artificial intelligence (AI)** to support **early, automated detection of respiratory diseases** through chest X-ray analysis. Developed in partnership with the **Ministry of Health (MINSAL)** and aligned with **UNDP’s 2025 AI for Sustainable Development Atlas**, the system aims to **empower medical professionals with AI tools** for faster and more accurate diagnosis—especially in resource-constrained settings.

---

## 🌍 Project Context

Respiratory diseases remain a major public health challenge in El Salvador. This initiative is designed to help:

- **Reduce diagnostic delays**
- **Alleviate the burden on radiologists**
- **Improve healthcare access in underserved areas**

Inspired by global best practices, similar AI models have shown diagnostic performance comparable to **six independent radiologists**. This offers a promising pathway for supporting national health systems.

---

## 🩻 Target Conditions

The AI system is trained to detect and classify the following diseases based on chest X-ray images:

1. **Pulmonary Tuberculosis**
2. **Pneumonia** (bacterial or viral)
3. **COVID-19**

These conditions were selected for their **high prevalence**, **radiographic visibility**, and **impact on the Salvadoran healthcare system**.

---

## 🧬 Technical Approach

- **Transfer Learning**: We employ a pre-trained **VGG16** Convolutional Neural Network (CNN) from ImageNet, fine-tuned for medical image classification.
- **Data Augmentation**: Techniques such as **rotation**, **cropping**, and **brightness adjustment** are applied to enhance model generalization without requiring massive datasets.
- **Model Deployment**: The AI model is built using **open-source tools** in Python to ensure **sustainability, transparency, and local capacity building**.

---

## 🧰 Libraries and Tools Used

```python
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
