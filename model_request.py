from pydantic import BaseModel, Field
from get_single_page import TopicRelevance, Analysis
import json

def determine_relevance(client, context, conversation_context, question):
    """
    Determine if the question is relevant to finance topics based on the context.
    """
    prompt = f"Context from documents:\n{context}{conversation_context}\n\nCurrent question: {question}"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config={
            "system_instruction": "You are a financial analyst assistant. Determine if the question is related to finance topics based on the provided document context.",
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": list[TopicRelevance]
        },
        contents=prompt
    )

    topic = json.loads(response.text)[0]["topic"]
    is_finance_related = json.loads(response.text)[0]["is_finance_related"]
    relevance_score = json.loads(response.text)[0]["relevance_score"]
    inverse_relevance_score = json.loads(response.text)[0]["inverse_relevance_score"]
    reasoning = json.loads(response.text)[0].get("reasoning")
    return topic, is_finance_related, relevance_score, reasoning, inverse_relevance_score

def general_overview_request(client, context, conversation_context, question):
    prompt = f"Context from documents:\n{context}{conversation_context}\n\nCurrent question: {question}"
    
    request_gemini = client.models.generate_content(
        model="gemini-2.0-flash",
        config={
            "system_instruction": "You are a financial analyst assistant. Answer questions based on the provided document context and conversation history. Be specific and reference the documents when possible. If the user is asking a follow-up question, consider the previous conversation context.",
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": list[Analysis]
        },
        contents=prompt
    )
    return request_gemini
