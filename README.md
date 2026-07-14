# 📊 Free-of-Cost LLM Evaluation CI/CD Pipeline

A modern, production-ready MLOps pipeline designed to run automated quality checks (LLM-as-a-judge) on every prompt change, model change, or knowledge base update. **Built entirely on free-tier services.**

## 🌟 Key Features

1. **Deterministic & Probabilistic Evaluators**: Grades answers on two key metrics using Gemini 1.5 Flash (via the free Gemini API):
   - **🎯 Faithfulness (Groundedness)**: Analyzes whether the generated response contains facts not supported by the retrieved context (hallucination detection).
   - **💬 Answer Relevancy**: Verifies if the answer directly responds to the prompt.
2. **Deterministic Keyword Scoring**: Quick keyword coverage tests.
3. **SLA Guardrails**: Automatically blocks a git pull request if:
   - Average faithfulness drops below **80%**.
   - Average relevancy drops below **80%**.
   - p95 Latency exceeds **3.0 seconds**.
4. **Interactive MLOps Dashboard**: A glassmorphic web dashboard hosted on **GitHub Pages** showing run trends and test-case metrics across git history.

---

## 🛠️ Free Tech Stack

- **Model Provider**: Google Gemini 1.5 Flash (via free API key)
- **Vector Database**: ChromaDB (In-Memory EphemeralClient)
- **Orchestration**: GitHub Actions (Free public runner minutes)
- **Metrics Store**: JSON file commits inside Git history (no Database costs)
- **Deployment**: GitHub Pages (Free static hosting)

---AQ.Ab8RN6KYbjN7wCvNp_TfyiZvJo0wSepJMD8rtRMkGVKBAh3zhw

## 📂 Project Structure

```
├── .github/workflows/
│   └── eval_workflow.yml     # CI/CD pipeline script
├── dashboard/
│   ├── index.html            # Glassmorphism front-end UI
│   ├── style.css             # Vanilla CSS UI styles
│   └── data/
│       └── run_history.json  # History logs containing all evaluation runs
├── kb/                       # Knowledge base folder containing source text files
│   └── sample.txt            # Sample product info text
├── eval_pipeline.py          # LLM-as-a-judge evaluation executor
├── rag_pipeline.py           # The RAG pipeline under test
├── golden_dataset.json       # 5 key QA test cases for evaluation
├── requirements.txt          # Python package requirements
└── .gitignore                # Git ignore configuration
```

---

## 🚀 Getting Started (Local Setup)

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Get a free API Key from [Google AI Studio](https://aistudio.google.com/) and set it in your terminal:
```bash
# Windows (CMD)
set GEMINI_API_KEY="your-key-here"

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-key-here"

# macOS/Linux
export GEMINI_API_KEY="your-key-here"
```

### 4. Run the Pipeline Locally
Run the pipeline to test it:
```bash
python eval_pipeline.py
```
This will run the evaluator and update `dashboard/data/run_history.json`.

### 5. Launch the Dashboard
Simply open `dashboard/index.html` in your browser to view the interactive charts!

---

## 🤖 CI/CD Integration & GitHub Pages Setup

### Step 1: Push to GitHub
Create a public repository on GitHub and push this code.

### Step 2: Add Secrets
Go to your repository **Settings > Secrets and Variables > Actions > New repository secret**:
- **Name**: `GEMINI_API_KEY`
- **Value**: *Your Google AI Studio API key*

### Step 3: Enable GitHub Pages
1. Go to repository **Settings > Pages**.
2. Under **Build and deployment > Source**, select **Deploy from a branch**.
3. Under **Branch**, select `gh-pages` and `/ (root)`, then click **Save**.
*(Note: The `gh-pages` branch will be automatically created by the workflow on your first push to main)*.

### Step 4: Allow GitHub Actions to Commit
Go to repository **Settings > Actions > General > Workflow permissions**:
- Select **Read and write permissions**.
- Click **Save**.

Now, whenever you push a change or make a pull request, the evaluation script will run automatically. If it passes, it will update the dashboard live on your GitHub Pages site!
