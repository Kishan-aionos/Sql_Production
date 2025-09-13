import re
import json
import asyncio
from typing import Optional, Dict
from groq import Groq
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from configs.config import GROQ_API_KEY, MODEL,LANGSMITH_API
from loggers.logger import llm_logger



client = Groq(api_key=GROQ_API_KEY)
# wrapped_client = wrap_openai(client)

SYSTEM_PROMPT = """
You are an assistant that converts natural language to SQL queries using ONLY the Northwind database schema for MySQL database.

CRITICAL: You MUST use MySQL syntax, not SQL Server or other dialects.

Tables:
- customers(customer_id, company_name, contact_name, country, region)
- categories(category_id, category_name)
- products(product_id, product_name, category_id, unit_price)
- orders(order_id, customer_id, order_date, ship_country, ship_region)
- order_details(order_id, product_id, unit_price, quantity, discount)

MySQL-SPECIFIC SYNTAX RULES:
1. Use LIMIT for row limiting: LIMIT 5 NOT TOP 5
2. Use IFNULL() instead of ISNULL()
3. Use backticks ` for quoting if needed: `order_date`
4. Use CONCAT() for string concatenation
5. Use DATE_FORMAT() for date formatting

Revenue calculation formula:
order_details.quantity * order_details.unit_price * (1 - IFNULL(order_details.discount, 0))

Chart selection rules:
1. If user specifies chart type → use that chart
2. Top N lists/comparisons → bar chart
3. Proportions/percentages → pie chart  
4. Time trends → line chart
5. Relationships → scatter plot
6. Default → table

Response format must be valid JSON:
{
    "sql": "SELECT ... OR null",
    "intent": "Historical|Forecasting|Unknown",
    "message": "Explanation if needed",
    "chart": "line|bar|pie|scatter|table|null"
}

Examples:
- "Top 5 products by revenue" → Use LIMIT 5, chart=bar
- "Top 5 products by revenue with pie chart" → Use LIMIT 5, chart=pie
- "Sales by month" → Use DATE_FORMAT(), chart=line
"""

@traceable(run_type="llm",name="Groq Completion")
async def llm_complete(system: str, user: str, temperature: float = 0.1) -> str:

    try:
        
        llm_logger.debug(f"LLM request - System: {system[:100]}..., User: {user[:100]}...")
        loop = asyncio.get_event_loop()
        
        def _llm_call():
            return client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
            )
        
        resp = await loop.run_in_executor(None, _llm_call)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        llm_logger.error(f"LLM API error: {e}")
        raise Exception(f"LLM API error: {str(e)}")

async def llm_complete_batch(system: str, user_messages: list, temperature: float = 0.1) -> list:
    """
    Process multiple LLM completions concurrently
    """
    tasks = []
    for user_message in user_messages:
        task = llm_complete(system, user_message, temperature)
        tasks.append(task)
    
    return await asyncio.gather(*tasks, return_exceptions=True)

async def llm_complete_with_retry(system: str, user: str, temperature: float = 0.1, 
                                 max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """
    LLM completion with retry mechanism for reliability
    """
    for attempt in range(max_retries):
        try:
            return await llm_complete(system, user, temperature)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s: {str(e)}")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from text that might contain extra content"""
    try:
        # First try to parse the whole text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to find JSON object within the text
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            try:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If still failing, try to find code blocks with JSON
        code_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_blocks:
            try:
                return json.loads(code_blocks[0])
            except json.JSONDecodeError:
                pass
        
        return None

async def extract_json_from_text_async(text: str) -> Optional[Dict]:
    """
    Async version of JSON extraction for CPU-bound processing
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_json_from_text, text)

def normalize_question(text: str) -> str:
    # Remove trailing punctuation (. ! ?)
    return re.sub(r"[.!?]+$", "", text.strip())

async def normalize_question_async(text: str) -> str:
    """
    Async version of question normalization
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, normalize_question, text)

def normalize_table_names(sql: str) -> str:
    # Fix common mistakes from LLM
    replacements = {
        "`order details`": "order_details",
        "order details": "order_details",
        "`Order Details`": "order_details",
        "[order details]": "order_details",
        "'order details'": "order_details"
    }
    for wrong, correct in replacements.items():
        sql = sql.replace(wrong, correct)
    return sql

async def normalize_table_names_async(sql: str) -> str:
    """
    Async version of table name normalization
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, normalize_table_names, sql)

async def determine_intent(question: str, system_prompt: str = SYSTEM_PROMPT) -> Dict[str, any]:
    """
    Determine the intent of a question using LLM asynchronously
    """
    try:
        raw_output = await llm_complete(system_prompt, question)
        intent_data = extract_json_from_text(raw_output)
        
        if not intent_data:
            return {"intent": "Unknown", "confidence": 0, "explanation": "Could not parse intent"}
        
        return {
            "intent": intent_data.get("intent", "Unknown"),
            "confidence": float(intent_data.get("confidence", 0)),
            "explanation": intent_data.get("explanation", ""),
            "sql": intent_data.get("sql"),
            "chart": intent_data.get("chart", "table")
        }
    except Exception as e:
        return {"intent": "Unknown", "confidence": 0, "explanation": f"Error determining intent: {str(e)}"}

async def process_nlq_to_sql(question: str) -> Dict[str, any]:
    """
    Complete async processing of natural language to SQL conversion
    """
    try:
        # Normalize question asynchronously
        normalized_question = await normalize_question_async(question)
        
        # Get LLM response asynchronously
        raw_output = await llm_complete(SYSTEM_PROMPT, normalized_question)
        
        # Extract JSON asynchronously
        data = await extract_json_from_text_async(raw_output)
        
        if not data:
            return {
                "success": False,
                "error": "Could not parse response from AI model",
                "intent": "Unknown"
            }
        
        # Normalize SQL asynchronously if present
        sql_query = data.get("sql")
        if sql_query:
            sql_query = await normalize_table_names_async(sql_query)
            sql_query = re.sub(r'```sql|```', '', sql_query).strip()
        
        return {
            "success": True,
            "sql": sql_query,
            "intent": data.get("intent", "Unknown"),
            "message": data.get("message", ""),
            "chart": data.get("chart", "table")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"NLQ processing error: {str(e)}",
            "intent": "Unknown"
        }

# For backward compatibility
async def llm_complete_async(system: str, user: str, temperature: float = 0.1) -> str:
    """
    Async alias for backward compatibility
    """
    return await llm_complete(system, user, temperature)