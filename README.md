# 🧠 VoiceLens – AI-Powered Image Descriptions for Accessibility

**VoiceLens** is an AI-driven accessibility tool that converts images into real-time audio descriptions for blind and visually impaired users. With a simple gesture like a triple-tap, it analyzes images and narrates meaningful captions using deep learning and advanced text-to-speech models.

---

## 🚨 Problem Statement

Visual content dominates the web and social media, but most screen readers can't interpret images meaningfully. This makes platforms like Instagram or Facebook less accessible for visually impaired users.

---

## 💡 Solution

VoiceLens bridges this gap by providing:

- ✅ Real-time **image captioning**
- 🔊 **Text-to-speech** narration of captions
- 🧠 Uses **ResNet50 + Transformer** for deep image understanding
- ✋ **Gesture-based interaction** (e.g., triple-tap, swipe)
- 🌐 Designed for **social media integration**

---

## 🎯 Objectives

- Build an AI pipeline with ResNet50 and Transformer
- Generate accurate captions and convert them to speech
- Enable real-time interaction via gestures
- Integrate seamlessly with social platforms
- Optimize based on usability testing and feedback

---

## ⚙️ Tech Stack

| Component         | Tool / Model Used         |
|------------------|---------------------------|
| Image Captioning | ResNet50 + Transformer    |
| Text-to-Speech   | SpeechT5 / gTTS           |
| UI & Backend     | Streamlit / Flask         |
| Datasets         | MS COCO, Flickr30k        |
| Deployment       | AWS / GCP                 |

---

## 🔄 Methodology

1. **Image Input** – User selects or triple-taps an image  
2. **Feature Extraction** – ResNet50 extracts visual features  
3. **Caption Generation** – Transformer model generates caption  
4. **Validation** – Caption is checked for accuracy  
5. **Speech Conversion** – Valid captions are converted to audio using TTS  
6. **Output** – Caption is read aloud along with image display  

---

## 🧱 System Architecture

- **Image Encoder** → ResNet50  
- **Caption Decoder** → Transformer  
- **Speech Generator** → SpeechT5 / gTTS  
- **Frontend** → Streamlit or Mobile Interface  
- **Backend** → Flask API (Cloud-hosted)

---

## 📊 Evaluation Metrics

- 🔵 **BLEU** – Caption accuracy  
- 🟡 **METEOR** – Sentence-level quality  
- 🔈 **MOS** – Voice clarity and naturalness  

--- 

---

> 🎯 *Empowering the visually impaired to hear what they cannot see.*
