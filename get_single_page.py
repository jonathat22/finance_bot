from docling.document_converter import DocumentConverter
from google import genai
from pydantic import BaseModel, HttpUrl
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
converter = DocumentConverter()

model="gemini-2.0-flash"

class PageRequest(BaseModel):
    url: HttpUrl
    page_number: int