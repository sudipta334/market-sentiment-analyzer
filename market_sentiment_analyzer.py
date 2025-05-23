import os
import requests
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

# Load environment variables from .env
load_dotenv()

# Optional debug check
assert os.getenv("AZURE_OPENAI_API_KEY"), "Missing AZURE_OPENAI_API_KEY"
assert os.getenv("LANGFUSE_PUBLIC_KEY"), "Missing LANGFUSE_PUBLIC_KEY"

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com"
)
langfuse_callback = LangfuseCallbackHandler()

# Static stock code lookup
stock_code_lookup = {
    "Apple Inc": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL"
}

# LLM initialization
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0
)

# Structured output parser
response_schemas = [
    ResponseSchema(name="company_name", description="Name of the company", type="string"),
    ResponseSchema(name="stock_code", description="Stock code of the company", type="string"),
    ResponseSchema(name="newsdesc", description="Summary of the news", type="string"),
    ResponseSchema(name="sentiment", description="Sentiment of the news (Positive/Negative/Neutral)", type="string"),
    ResponseSchema(name="people_names", description="List of people mentioned", type="list"),
    ResponseSchema(name="places_names", description="List of places mentioned", type="list"),
    ResponseSchema(name="other_companies_referred", description="List of other companies mentioned", type="list"),
    ResponseSchema(name="related_industries", description="List of related industries", type="list"),
    ResponseSchema(name="market_implications", description="Market implications of the news", type="string"),
    ResponseSchema(name="confidence_score", description="Confidence score of the sentiment", type="float")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Prompt template
sentiment_prompt_template = PromptTemplate(
    input_variables=["company_name", "stock_code", "news"],
    template="""
    You are a financial analyst. Analyze the following news about {company_name} (stock code: {stock_code}) and provide a structured sentiment profile.

    News: {news}

    - Classify the sentiment as Positive, Negative, or Neutral.
    - Extract named entities (people, places, other companies).
    - Identify related industries and market implications.
    - Provide a confidence score for your sentiment analysis (between 0 and 1).

    Format your response according to the following schema:
    {format_instructions}
    """,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# LLM Chain (consider refactoring to RunnableSequence later)
sentiment_chain = LLMChain(
    llm=llm,
    prompt=sentiment_prompt_template,
    output_parser=output_parser,
    output_key="sentiment_result",
    callbacks=[langfuse_callback]
)

# Main analyzer class
class MarketSentimentAnalyzer:
    def __init__(self):
        self.stock_code_lookup = stock_code_lookup
        self.sentiment_chain = sentiment_chain

    def get_stock_code(self, company_name):
        return self.stock_code_lookup.get(company_name, "Unknown")

    def fetch_news(self, stock_code):
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={stock_code}&esCount=1&newsCount=5"
            headers = {
                "User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0")
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            news_items = data.get("news", [])
            if not news_items:
                return "No news found."
            news_text = "\n".join([item.get("title", "") for item in news_items[:5]])
            return news_text if news_text else "No news found."
        except Exception as e:
            return f"Error fetching news: {str(e)}"

    def analyze_sentiment(self, company_name, stock_code, news):
        return self.sentiment_chain.run(
            company_name=company_name,
            stock_code=stock_code,
            news=news
        )

    def run(self, company_name):
        stock_code = self.get_stock_code(company_name)
        if stock_code == "Unknown":
            return {"error": f"Stock code for {company_name} not found."}
        news = self.fetch_news(stock_code)
        result = self.analyze_sentiment(company_name, stock_code, news)
        return result


# Entry point
if __name__ == "__main__":
    analyzer = MarketSentimentAnalyzer()
    company_name = "Microsoft"
    result = analyzer.run(company_name)
    print(result)

