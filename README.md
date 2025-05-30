
# Real-Time Speech-to-Speech Translation System

## 📌 Overview

This project implements a **real-time speech-to-speech translation system** using **IoT technology**. The system captures spoken input, transcribes it to text, translates it into the target language, and finally converts it back to speech, delivering output through a speaker. It aims to **bridge language barriers** in domains like education, healthcare, travel, and business.

## 🎯 Key Features

- **Real-time translation**
- **Supports 30 languages** (e.g., English, Hindi, German, French)
- **Achieved 86% overall accuracy**
- **Portable and cost-effective** using Raspberry Pi
- **Offline-ready (future enhancement)**

## 🛠 System Components

### 🔧 Hardware
- **Raspberry Pi** – Core processing unit
- **Microphone** – Captures user speech
- **Speaker** – Outputs translated speech
- **Power Supply** – Supports mobility and reliability
  
<img src="https://github.com/user-attachments/assets/9dcefbe2-3cda-4849-bd31-6d731c30af30" width="300"/> <img src="https://github.com/user-attachments/assets/eb4df2d8-761c-4cb8-9847-f382383d9531" width="300"/> <img src="https://github.com/user-attachments/assets/3d9c5afe-ed82-4ad4-ad9e-d9af87e11b50" width="300"/>

### 🧠 Software Workflow
1. **Speech Recognition** (via `speech_recognition`)
2. **Text Translation** (via `googletrans`)
3. **Text-to-Speech** (via `gTTS`)
4. **Real-time Output Playback** (via `pygame`)

![WhatsApp Image 2025-05-24 at 13 52 30_b21e91b0](https://github.com/user-attachments/assets/c289c9b8-6d13-4197-8fe0-d615e1327494)

## 📊 Accuracy Evaluation

- **Speech Recognition Accuracy**: Assessed by comparing recognized text to known original speech
- **Translation Accuracy**: Measured using BLEU score with smoothing
- **Combined Accuracy Factor**: Calculated using a weighted combination of both speech and translation accuracy
- **Final Accuracy**:  
  🔹 **Speech-to-Speech Accuracy**: **86%**

## 📈 Performance Metrics

| Metric                    | Value                    |
|---------------------------|--------------------------|
| Translation Accuracy      | 86% overall              |
| Languages Supported       | 30+                      |
| Translation Latency       | Low (real-time capable)  |
| Practical Use Cases       | Travel, Education, Work  |
| Speech + Translation BLEU | Used for evaluation      |


<img src="https://github.com/user-attachments/assets/89d42e23-5fa2-4faf-9b02-2cf15e9cc5d3" width="300"/> <img src="https://github.com/user-attachments/assets/94d6d7cf-826e-4ff7-8690-700b32a60957" width="300"/> <img src="https://github.com/user-attachments/assets/8863bd67-ba4f-4775-b049-4e132e147180" width="300"/>

## 📈 Visualizations
- **Accuracy Factor Plot**: Shows change in accuracy over multiple translations
- **Recognition vs Translation Time Correlation**: Identifies performance trends

## 🚀 Setup Instructions

1. Install Python dependencies:
   ```bash
   pip install speechrecognition googletrans==4.0.0-rc1 gtts pygame nltk matplotlib transformers
   ```

2. Connect hardware:
   - Microphone & speaker to Raspberry Pi
   - Ensure power supply is stable

3. Run the main script:
   ```bash
   python translator.py
   ```

4. Speak a phrase and enter a **two-letter language code** (e.g., `fr`, `de`, `es`) when prompted.

## ⚙️ Optimization Strategies

- **Noise Cancellation**: Implemented echo cancellation and microphone array filtering
- **Hardware Efficiency**: Sampling, compression, and optimized Raspberry Pi use
- **Accuracy Improvements**: Uses attention-based NMT models for higher fidelity

## 🔮 Future Enhancements

- Offline translation capability for rural/remote deployment
- Support for dialect detection and regional languages
- Mobile and wearable device integration
- Advanced noise handling and better contextual understanding
- Stronger encryption for data privacy


https://github.com/user-attachments/assets/5c7a6e0c-0d35-47ec-8fd3-80e67c1b6003


