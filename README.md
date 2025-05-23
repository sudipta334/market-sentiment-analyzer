# Real-Time Market Sentiment Analyzer

This project implements a LangChain-powered pipeline to analyze market sentiment for a given company using Yahoo Finance news and Azure OpenAI GPT-4o. The pipeline fetches news for a specified company, analyzes the sentiment using Azure OpenAI, and outputs a structured JSON response. Langfuse is integrated for tracing and monitoring.
# Tech Stack

- **LLM:** Azure OpenAI (gpt-4o-mini)
- **Frameworks:** LangChain, Langfuse
- **Data Source:** Yahoo Finance
- **Language:** Python 3.12+
- **Environment Management:** `venv` + `.env`
## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/market-sentiment-analyzer.git
   cd market-sentiment-analyzer
   python market-sentiment-analyzer.py