
# 🌾 Rice Crop Nutrient Deficiency Detection & Chat Assistant

This app helps farmers and agricultural researchers identify **nutrient deficiencies in rice crops** through image analysis, followed by an **interactive chat assistant** that provides detailed instructions, explanations, and support.

## 🧠 What It Does

1. **Upload Crop Image**
   The user uploads an image of a rice plant showing visible symptoms.

2. **Deficiency Detection (YOLOv8)**
   A YOLO-based deep learning model detects the **type of nutrient deficiency** (e.g., Nitrogen, Phosphorus, Potassium).

3. **Intelligent Chat Assistant (LLaMA 3 via Groq)**
   After detection, an AI assistant explains the deficiency, suggests remedies, and answers follow-up questions in a natural chat interface.

## 🚀 Key Features

* 📷 **Fast and Accurate Detection** using a YOLO model fine-tuned for rice crop imagery.
* 💬 **Conversational AI** powered by LLaMA 3 (70B) via Groq for real-time, helpful interactions.
* 🧑‍🌾 **Farmer-Friendly** instructions in simple language.
* 🌐 **Streamlit Web Interface** for easy access and usability.

## 🛠️ Tech Stack

| Component       | Tool/Framework           |
| --------------- | ------------------------ |
| Detection Model | YOLOv8                   |
| Chat Assistant  | LLaMA 3 70B via Groq API |
| UI Framework    | Streamlit                |
| Backend Runtime | Python                   |


## 📦 Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/ikramali585/Nutrient-deficiency-chatbot.git
   cd Nutrient-deficiency-chatbot
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up Groq API key in .env:

   ```bash
   GROQ_API_KEY=your_key_here
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

## 📜 License

MIT License

