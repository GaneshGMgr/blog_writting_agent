# ✍️ Blog Writing Agent (LangGraph + Streamlit)

An advanced **AI-powered Blog Writing Agent** built using **LangGraph, LangChain, and Streamlit**.  
This system automatically generates **well-structured, research-backed blog posts with optional AI-generated images**.

---

## 🚀 Features

- 🧠 **Autonomous Blog Generation**
  - Input a topic → get a complete blog post

- 🔀 **Smart Routing (Closed / Hybrid / Open Book)**
  - Decides whether research is needed
  - Adapts writing style accordingly

- 🔎 **Automated Research (Tavily API)**
  - Fetches real-time web data for accuracy

- 🧩 **Structured Blog Planning**
  - Generates:
    - Title
    - Sections
    - Goals
    - Word targets

- ⚡ **Parallel Section Writing**
  - Each section generated independently (LangGraph workers)

- 🖼️ **AI Image Generation**
  - Automatically inserts diagrams/images
  - Uses Gemini image generation

- 💬 **Streaming UI**
  - Real-time progress tracking in Streamlit

- 📂 **Download Options**
  - Markdown file
  - Full bundle (Markdown + images)

- 🕘 **Past Blog Loader**
  - Load previously generated `.md` files

---

## 🖼️ Output / Demo

### Blog Preview
![Blog Output](Output/result1.png)

---

## 🏗️ Architecture Overview

    User Input (Topic)
            ↓
        Router Node
   (decides research mode)
            ↓
     ┌───────────────┐
     │ Research Node │ (optional)
     └───────────────┘
            ↓
     Orchestrator (Plan)
            ↓
   Parallel Workers (Sections)
            ↓
   Reducer Pipeline:
      - Merge Content
      - Decide Images
      - Generate Images
            ↓
       Final Blog (Markdown)

---

## 📂 Project Structure

    .
    ├── app.py                         # Streamlit UI
    ├── blog_writing_agent_backend.py # LangGraph pipeline
    ├── images/                        # Generated images
    ├── Output/
    │   └── result1.png
    ├── .env
    └── README.md

---

## ⚙️ Installation

    git clone https://github.com/your-username/blog-writing-agent.git
    cd blog-writing-agent

    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows

    pip install -r requirements.txt

---

## 🔑 Environment Variables

Create a `.env` file:

    OPENAI_API_KEY=your_openai_api_key
    OPENAI_BASE_URL=your_base_url
    REASONING_MODEL=your_model_name
    TAVILY_API_KEY=your_tavily_key
    GOOGLE_API_KEY=your_gemini_key

---

## ▶️ Run the App

    streamlit run app.py

---

## 🧠 How It Works

### 1. Router
- Decides:
  - `closed_book` → no research
  - `hybrid` → partial research
  - `open_book` → full research

### 2. Research (Optional)
- Uses Tavily API
- Filters relevant and recent sources

### 3. Orchestrator
- Generates structured blog plan
- Defines sections and writing strategy

### 4. Workers (Parallel Execution)
- Each section written independently
- Ensures scalability and speed

### 5. Reducer Pipeline
- Merge all sections
- Decide if images are needed
- Generate and insert images

---

## 🛠️ Core Components

- **Router Node** → decides research need  
- **Research Node** → fetches web data  
- **Orchestrator** → builds blog structure  
- **Worker Nodes** → generate sections  
- **Reducer Subgraph**:
  - merge_content
  - decide_images
  - generate_and_place_images  

---

## 🖼️ Image Generation Flow

    Markdown → Insert Placeholders
            ↓
    Generate Image Prompts
            ↓
    Gemini Image API
            ↓
    Save to /images
            ↓
    Replace placeholders with images

---

## 📥 Downloads

- ✅ Markdown file
- 📦 ZIP bundle (Markdown + images)
- 🖼️ Images-only ZIP

---

## 🧩 Technologies Used

- **Frontend:** Streamlit  
- **Orchestration:** LangGraph  
- **LLM:** OpenAI-compatible models  
- **Search:** Tavily API  
- **Images:** Google Gemini API  
- **Data Handling:** Pandas  
- **Storage:** Local files  

---

## ⚠️ Limitations

- Requires API keys (OpenAI, Tavily, Gemini)
- Image generation depends on external API
- No database (file-based storage)
- Large blogs may take time

---

## 🔮 Future Improvements

- Persistent blog storage (DB)
- Multi-user support
- SEO optimization
- Blog export (HTML / PDF)
- CMS integration (WordPress, Notion)

---

## 🤝 Contributing

Pull requests are welcome.  
For major changes, please open an issue first.

---

## 📜 License

MIT License

---

## 💡 Acknowledgements

- LangGraph & LangChain teams  
- OpenAI & Gemini APIs  
- Streamlit for UI  