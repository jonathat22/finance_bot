from docling.document_converter import DocumentConverter
from google import genai
from pydantic import BaseModel, HttpUrl, Field
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
converter = DocumentConverter()

model="gemini-2.0-flash"

class PageRequest(BaseModel):
    url: HttpUrl


class Summary(BaseModel):
    summary: str

class TopicRelevance(BaseModel):
    """
    A model to represent the relevance of a topic to a given query.
    """
    topic: str = Field(description="The topic being evaluated for relevance.")
    is_finance_related: bool = Field(
        description="Indicates whether the topic is related to finance."
        )
    relevance_score: float = Field(
        description="A score indicating how relevant the topic is to the query, ranging from 0.0 to 1.0."
    )
    inverse_relevance_score: float = Field(
        description="1 minus the relevance score"
    )
    reasoning: str = Field(description="Explanation of why the topic is considered relevant or not.")

class Analysis(BaseModel):
    """
    A model to provide both general and specific analysis of the company.
    """
    answer: str = Field(
        description="The answer to the user's question, providing insights based on the provided context."
    )
    revenue: float = Field(
        description="The revenue or total net sales of the company."
    )
    cost_of_goods_sold: float = Field(
        description="The cost of goods sold (COGS) or cost of revenue for the company."
    )
    gross_profit: float = Field(
        description="The gross profit, calculated as revenue minus cost of goods sold."
    )
    operating_expenses: float = Field(
        description="Total operating expenses, including research & development, selling, general and administrative expenses."
    )
    operating_income: float = Field(
        description="Operating income, calculated as gross profit minus operating expenses."
    )


url = "https://www.apple.com/newsroom/pdfs/fy2025-q4/FY25_Q4_Consolidated_Financial_Statements.pdf"
page_request = PageRequest(url=url)
document_text = converter.convert(str(page_request.url))
content = document_text.document.export_to_markdown()


response_gemini = client.models.generate_content(
        model="gemini-2.0-flash",
        config={
            "system_instruction": "You are an equity research analyst. Briefly summarize the financial results presented in this document.",
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": Summary
        },
        contents=f"Content from the web page:\n{content}"
    )

result = response_gemini.to_model(list[Summary])
print("Summary of the web page content:")
print(result[0].summary)
