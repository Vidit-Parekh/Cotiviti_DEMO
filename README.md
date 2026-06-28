# ClinicalAI Insight

**An Explainable Multimodal Clinical Document Intelligence Platform**

> Cotiviti Intern Assessment | Vidit Parekh | University of Cincinnati | M.Eng. Computer Science

---

## Overview

ClinicalAI Insight is a proof-of-concept platform demonstrating how modern AI technologies can work together to accelerate clinical document review for healthcare payers and analytics organizations like Cotiviti.

The core thesis: healthcare AI is not evolving toward bigger chatbots. It is evolving toward **trustworthy clinical intelligence systems** that keep humans in control of all final decisions.

---

## Features

| Feature | Description |
|---|---|
| PDF and TXT ingestion | Upload clinical records directly; PyMuPDF handles text extraction |
| AI Clinical Summary | 2 to 3 sentence LLM-generated summary of the document |
| Diagnosis Extraction | Named entity recognition for diagnoses with ICD-10 code suggestions |
| Medication Extraction | Identifies medications with dose, route, frequency, and status |
| Procedure Extraction | Identifies procedures with CPT code suggestions |
| Confidence Scoring | Calibrated confidence scores on all extracted entities |
| Evidence Panel | Shows the exact source text that supports each extracted entity |
| Payment Integrity Review | Flags documentation gaps relevant to coding and billing |
| Q&A Assistant | RAG-grounded question answering over the uploaded document |

---

## Technology Stack

- **LLM:** Llama 3.3 70B via Groq free API (or Ollama local fallback)
- **Document Ingestion:** PyMuPDF (OCR and PDF parsing)
- **NLP Pipeline:** LLM-based clinical named entity recognition
- **RAG Architecture:** Document-grounded prompting to reduce hallucinations
- **Frontend:** Streamlit
- **Language:** Python 3.12

---

## AI Pipeline Architecture

```
Clinical Document (PDF or TXT)
         |
         v
  Text Extraction (PyMuPDF)
         |
         v
  Clinical NLP (LLM entity recognition)
         |
         v
  RAG Grounding (document-anchored context window)
         |
         v
  LLM Analysis (Llama 3.3 70B via Groq or Ollama)
         |
         v
  Structured Output (diagnoses, medications, procedures)
         |
         v
  Explainable Insights with Evidence Panel
         |
         v
  Payment Integrity Review
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A free Groq API key (get one at [console.groq.com](https://console.groq.com)) **OR** [Ollama](https://ollama.com) installed locally

### Installation

```bash
git clone https://github.com/Vidit-Parekh/ClinicalAIInsight.git
cd ClinicalAIInsight
pip install -r requirements.txt
```

### Set your API key

```bash
# Option A: Groq free API (recommended, no GPU needed)
export GROQ_API_KEY=your_key_here        # Mac/Linux
set GROQ_API_KEY=your_key_here           # Windows

# Option B: Ollama local (no API key needed)
# Install Ollama, then run: ollama pull deepseek-coder
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Project Structure

> **The PowerPoint presentation, written report, and video recording are located in the `assets/` folder.**

```
ClinicalAIInsight/
|
|-- app.py                  # Main Streamlit application
|-- requirements.txt        # Python dependencies
|-- README.md               # This file
|
|-- assets/                  # Presentation, report and recording files
|   |-- presentation/
|   |   |-- Clinical_NLP_Presentation_Vidit_Parekh.pptx
|   |-- report/
|   |   |-- Clinical_NLP_Report_Vidit_Parekh.docx
|   |-- recording/
|       |-- demo_recording.mp4
|
|-- data/
|   |-- sample_record.txt   # Sample clinical note for demo
|
|-- src/
    |-- ingestion/
    |   |-- pdf.py          # PDF text extraction module
    |   |-- ocr.py          # OCR pipeline
    |
    |-- nlp/
    |   |-- entities.py     # Clinical entity extraction
    |   |-- summarizer.py   # Clinical summarization
    |
    |-- rag/
    |   |-- embeddings.py   # Document embedding pipeline
    |   |-- vectorstore.py  # In-memory vector index
    |
    |-- llm/
    |   |-- prompts.py      # System prompts and templates
    |
    |-- report/
        |-- generator.py    # Exportable report generation
```

---

## Strategic Context

This project was built as part of the Cotiviti Intern Assessment, Topic 1: Clinical Natural Language Technology for Health Care.

The platform directly illustrates three strategic recommendations from the accompanying white paper:

1. **AI-Powered Medical Coding Auditor** demonstrated via ICD-10 and CPT code extraction with evidence grounding
2. **Multimodal Prior Authorization Intelligence** demonstrated via PDF ingestion, NLP, and document Q&A
3. **Hallucination-Mitigated Clinical NLP Platform** demonstrated via confidence scoring, evidence panels, and RAG-grounded responses

---

## Disclaimer

ClinicalAI Insight is a prototype for demonstration purposes only. All outputs including clinical summaries, entity extractions, ICD-10 code suggestions, CPT code suggestions, and payment integrity observations are AI-generated illustrations. They are not validated medical outputs and must not be used for any clinical, billing, coding, or payment integrity decisions. All suggestions require qualified human review before any operational use.

---

## Author

**Vidit Parekh**
University of Cincinnati | M.Eng. Computer Science
[LinkedIn](https://linkedin.com/in/vidit-parekh26) | [GitHub](https://github.com/Vidit-Parekh) | [Portfolio](https://vidit-parekh.github.io)