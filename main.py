import os
import uvicorn
import asyncio
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prophet import Prophet
from logger import api_logger, logger

from config import MODEL_PATH
from database import get_connection, run_sql_async, get_sales_data_async, get_table_stats_async
from llm_utils import (
    SYSTEM_PROMPT, 
    llm_complete_async, 
    extract_json_from_text, 
    normalize_question, 
    normalize_table_names,
    process_nlq_to_sql
)
from forecast_utils import (
    train_forecast_model_async, 
    generate_forecast_async, 
    generate_forecast_summary_async
)



from pydantic import BaseModel

class NLQRequest(BaseModel):
    question: str

class ForecastRequest(BaseModel):
    periods: int = 30

# Initialize FastAPI app
app = FastAPI(title="Northwind NL→SQL + Forecast API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Northwind API application starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Northwind API application shutting down")


# ================= Health Check =================
@app.get("/health")
async def health_check():
    try:
        connection = await get_connection()
        async with connection.cursor() as cur:
            await cur.execute("SELECT 1")
        connection.close()
        api_logger.info("Health check successful - database connected")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

# ================= Train Model Endpoint =================
# ================= Train Model Endpoint =================
@app.post("/train_forecast")
async def train_forecast():
    api_logger.info("Train forecast endpoint called")
    result = await train_forecast_model_async(MODEL_PATH)
    
    if not result["success"]:
        api_logger.warning(f"Forecast training failed: {result['message']}")
        # Get debug info about the database asynchronously
        try:
            stats = await get_table_stats_async()
            debug_info = {
                "orders_count": stats.get("orders", 0),
                "order_details_count": stats.get("order_details", 0),
                "message": "Check if orders and order_details tables have data"
            }
        except Exception as e:
            debug_info = {"error": str(e)}
        
        raise HTTPException(
            status_code=400,
            detail={
                "message": result["message"],
                "debug_info": debug_info
            }
        )
    
    api_logger.info("Forecast model trained successfully")
    return result


# ================= Debug Endpoint =================
@app.get("/debug/sales-data")
async def debug_sales_data():
    """Debug endpoint to check if sales data fetching works"""
    try:
        from database import get_sales_data_async
        data = await get_sales_data_async()
        return {
            "success": True,
            "data_count": len(data) if data else 0,
            "sample_data": data[:3] if data else []
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ================= Forecast Endpoint =================
@app.get("/forecast")
async def forecast_sales(periods: int = 30):
    try:
        forecast_data = await generate_forecast_async(periods, MODEL_PATH)
        return forecast_data
    except Exception as e:
        if "No trained model found" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

# ================= Ask Endpoint =================
@app.post("/ask")
async def ask_question(body: NLQRequest):
    api_logger.info(f"Ask endpoint called with question: {body.question}")
    
    # Step 1: Process NLQ to SQL asynchronously
    nlq_result = await process_nlq_to_sql(body.question)
    
    if not nlq_result["success"]:
        api_logger.error(f"NLQ processing failed: {nlq_result.get('error')}")
        return {
            "question": body.question,
            "intent": "Unknown",
            "sql": None,
            "message": nlq_result["error"],
            "result": None
        }

    sql_query = nlq_result.get("sql")
    intent = nlq_result.get("intent", "Unknown")
    message = nlq_result.get("message", "")
    chart = nlq_result.get("chart", "table")

    api_logger.info(f"Question intent: {intent}, SQL generated: {sql_query is not None}")

    # Step 2: execute SQL if Historical
    result = None
    forecast_summary = None
    
    if intent == "Historical" and sql_query:
        try:
            api_logger.debug(f"Executing historical SQL query: {sql_query}")
            result = await run_sql_async(sql_query)
            api_logger.info(f"SQL query executed successfully, returned {len(result.get('rows', []))} rows")
        except Exception as e:
            api_logger.error(f"SQL execution error: {e}")
            result = {"message": str(e)}

    # Step 3: forecasting logic
    elif intent == "Forecasting":
        try:
            api_logger.info("Generating forecast for question")
            forecast_data = await generate_forecast_async(30, MODEL_PATH)
            forecast_summary = await generate_forecast_summary_async(body.question, forecast_data)
            result = forecast_data
            api_logger.info(f"Forecast generated successfully: {len(forecast_data)} periods")
        except Exception as e:
            if "No trained model found" in str(e):
                api_logger.warning("No trained forecast model found")
                result = {"message": "No trained forecast model found. Please train the model first using /train_forecast endpoint."}
            else:
                api_logger.error(f"Forecast generation error: {e}")
                result = {"message": f"Forecast error: {str(e)}"}

    api_logger.info(f"Question processing completed: intent={intent}")
    
    return {
        "question": body.question,
        "intent": intent,
        "sql": sql_query,
        "message": message,
        "result": result,
        "forecast_summary": forecast_summary,
        "chart": chart
    }

# ================= Get Table Stats Endpoint =================
@app.get("/debug/stats")
async def get_debug_stats():
    """Endpoint for debugging database table statistics"""
    try:
        stats = await get_table_stats_async()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# ================= Run App =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)