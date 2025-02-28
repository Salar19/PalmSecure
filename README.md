# PalmSecure - Revolutionizing Transport with Biometric Precision

## Overview
**PalmSecure** is a state-of-the-art palmprint recognition system designed specifically for the transport industry. It leverages the **Comprehensive Competition Network (CCNet)** model to provide a **non-contact, hygienic, and highly accurate biometric verification solution**. The system enhances security, operational efficiency, and user convenience, making it ideal for high-traffic environments like transportation hubs.

---

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [CCNet Architecture](#ccnet-architecture)
4. [Features](#features)
5. [Datasets](#datasets)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [Acknowledgments](#acknowledgments)
12. [Contributors](#contributors)

---

## Abstract
Traditional biometric systems, such as fingerprint and facial recognition, often face challenges related to environmental sensitivity, hygiene concerns, and limited accuracy. **PalmSecure** introduces a novel palmprint verification system using the **Comprehensive Competition Network (CCNet)**, which integrates **spatial, channel, and multi-order competition mechanisms** to extract robust and discriminative features from palmprint images. 

The system is designed to address the limitations of existing biometric solutions, offering **non-contact operation, high accuracy, and scalability** for various industries, including transportation, banking, and healthcare. Extensive experimental results demonstrate that **PalmSecure** achieves **state-of-the-art performance** across multiple public datasets, making it a transformative solution for biometric authentication.

---

## Introduction
The transport industry is a critical component of modern society, facilitating the movement of millions of people daily. However, it faces significant challenges related to security, operational efficiency, and passenger safety. Traditional biometric systems, such as fingerprint and facial recognition, have proven inadequate due to their sensitivity to environmental factors, hygiene concerns, and limited accuracy in diverse conditions.

**Palmprint recognition** has emerged as a promising alternative, leveraging the unique and stable features of the human palm, such as ridges, wrinkles, and minutiae. This biometric modality offers several advantages, including **non-contact operation**, resilience to environmental variations, and high resistance to forgery. These characteristics make it an ideal candidate for enhancing security in the transport sector.

The **Comprehensive Competition Network (CCNet)** is a deep learning-based framework that revolutionizes palmprint recognition by integrating **spatial, channel, and multi-order competition mechanisms**. CCNet enhances recognition accuracy by combining channel and spatial information, offering a robust and scalable solution for biometric identification. This project, titled **PalmSecure**, aims to develop a cutting-edge palmprint recognition system tailored to the transport sector, leveraging CCNet to provide a **user-friendly, efficient, and scalable identity verification solution**.

---

## CCNet Architecture
The **Comprehensive Competition Network (CCNet)** is designed to enhance palmprint recognition by integrating **spatial, channel, and multi-order competition mechanisms**. The architecture comprises the following components:

- **Learnable Gabor Filters:** The model employs learnable Gabor filters in its texture extraction layers, enabling automatic adaptation to varying input features.
- **Spatial Competition Module:** Extracts spatial competition features by analyzing the relationships between different regions of the palmprint image.
- **Channel Competition Module:** Extracts channel-based competitive features, determining the dominant texture responses along specific feature channels.
- **Multi-Order Competition Module:** Captures multi-scale and higher-order texture features to improve the robustness and discrimination of the recognition process.
- **Comprehensive Competition Mechanism:** Integrates spatial, channel, and multi-order competition mechanisms into a unified feature extraction framework.

The network's architecture ensures efficient feature extraction and improved recognition accuracy by leveraging the complementary nature of these components.

---

## Features
- **Non-contact biometric verification:** Ensures hygienic operation, especially critical in high-traffic environments like transportation hubs.
- **Comprehensive Competition Network (CCNet):** Leverages spatial, channel, and multi-order competition mechanisms for high-accuracy palmprint recognition.
- **Multi-order texture feature extraction:** Captures higher-order texture features for superior discrimination.
- **Scalable architecture:** Designed for seamless integration with existing infrastructure, making it suitable for cross-industry applications.
- **Robust performance:** Achieves state-of-the-art accuracy across multiple datasets, including Tongji, CASIA, and COEP.

---

## Datasets
The **PalmSecure** system was trained and evaluated on the following datasets:

1. **Tongji Contactless Palmprint Dataset:**
   - Contains 12,000 images from 300 individuals.
   - Collected in a contactless setup, making it ideal for hygienic applications.

2. **CASIA Palmprint Image Database:**
   - Consists of 5,502 grayscale images from 312 individuals.
   - Includes variations in lighting and hand positioning.

3. **COEP Palmprint Dataset:**
   - Contains 1,344 high-resolution images from 168 individuals.
   - Captured under controlled conditions with variations in hand orientation.

4. **Locally Collected Dataset:**
   - Comprises 240 palm images from 30 users.
   - Collected using personal mobile phones, introducing variability in image quality and environmental conditions.

---

## Data Preprocessing
- **Image Resizing:** All images were resized to 224x224 pixels.
- **Normalization:** Pixel intensity values were normalized to [0, 1].
- **Data Augmentation:** Rotation, scaling, flipping, and illumination adjustments.
- **Noise Reduction:** Median filtering was used to enhance palmprint features.

---

## Model Training
- **Optimizer:** Adam (learning rate 0.0005, decayed by 0.1 every 10 epochs).
- **Loss Function:** Hybrid of cross-entropy and contrastive loss.
- **Batch Size:** 256.
- **Training Environment:** NVIDIA RTX 3090 GPU, PyTorch framework.

### Training Command:

- **batch\_size**: Default `1024`
- **epoch\_num**: Default `3000`
- **temp**: Contrastive loss temperature, default `0.07`
- **weight1**: Cross-entropy loss weight, default `0.8`
- **weight2**: Contrastive loss weight, default `0.2`
- **com\_weight**: Traditional competition mechanism weight, default `0.8`
- **lr**: Learning rate, default `0.001`
- **redstep**: Learning scheduler step size, default `500`

```bash
python train.py --id_num xxxx --train_set_file xxxx --test_set_file xxxx --des_path xxxx --path_rst xxxx
```

---

## Evaluation
**Metrics Used:**
- Equal Error Rate (EER)
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Accuracy

### Key Results:
- **Tongji Dataset:** 0.4% EER, 94% accuracy.
- **CASIA Dataset:** 20% EER, 84% accuracy.
- **COEP Dataset:** 5.77% EER, 96% accuracy.
- **Local Dataset:** 0.1% EER, 99% accuracy.

---

## Conclusion
PalmSecure, powered by CCNet, delivers a **hygienic, scalable, and high-accuracy** palmprint recognition system, ideal for transport security. Despite challenges with dataset variability, it demonstrates **state-of-the-art** performance and real-world applicability.

---

## Acknowledgments
- We extend our gratitude to our internal supervisor, **Dr. Muhammad Atif Tahir**, for his invaluable guidance and contributions to this project. 
- We also acknowledge **Zi-YuanYang** for the development of CCNet. The original CCNet repository can be found at [CCNet GitHub Repository](https://github.com/Zi-YuanYang/CCNet.git).

---

## Contributors
- **Muhammad Salar**
- **Muhammad Hamza Gova**
- **Muhammad Talha Bilal**

For inquiries or contributions, visit [GitHub Repository](https://github.com/Salar19).
