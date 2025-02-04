# INDIAN-GST-chatbot-with-Invoice-analysis

# RAG-Based GST Analysis Agent

## Overview

This project is a Retrieval-Augmented Generation (RAG) based agent for GST (Goods and Services Tax) analysis. The system is designed to extract, process, and analyze GST-related data from various sources, including PDFs, images, and text inputs. It leverages large language models and advanced document parsing techniques to provide meaningful insights.

### Key Features

* **RAG-Based Architecture:** Combines retrieval and generation for accurate GST analysis.
* **Multi-Modal Input:** Supports text, PDFs, and images.
* **Advanced OCR & Table Extraction:** Extracts structured data from scanned invoices and GST documents.
* **Integration with O1-mini Model:** Utilizes the O1-mini model for enhanced response generation.

---

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

* Python 3.8+
* Pip
* Virtual Environment (recommended)

### Setup

### git clone 

### [https://github.com/your-repo/gst-analysis-agent.git](https://github.com/your-repo/gst-analysis-agent.git)

### cd gst-analysis-agent

- python -m venv botenv
- source botenv/bin/activate  # On Windows, use: botenv\Scripts\activate
- pip install -r requirements.txt
- setup api key in .env (open api key)

### how to run

- python app.py

### API Endpoints

1. **Upload Data:**
   * Endpoint: `<span>/UploadData</span>`
   * Method: `<span>POST</span>`
   * Supports: Text, PDFs, Images
2. **PDF Processing:**
   * Uses `<span>unstructured.partition.pdf</span>` for structured data extraction.
   * Extracts tables and text separately.
3. **Image Processing:**
   * Uses `<span>pytesseract</span>` for OCR-based text extraction.
   * Object detection using `<span>torchvision</span>`.
4. **Integration with O1-mini:**
   * The system leverages the `<span>O1-mini</span>` model for intelligent text generation and reasoning over retrieved GST data.
