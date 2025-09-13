import pickle
import pandas as pd
from prophet import Prophet
import asyncio
from typing import List, Dict, Any, Optional

# Import the database function with a different name to avoid conflict
from database_detail.database import get_sales_data_async as fetch_sales_data_from_db
from llms.llm_utils import llm_complete_async
from configs.config import MODEL
from loggers.logger import forecast_logger


async def get_sales_data_for_forecasting():
    """
    Fetch daily total sales from orders + order_details for forecasting.
    """
    try:
        forecast_logger.info("Fetching sales data for forecasting")
        # Use the async database function with the renamed import
        result = await fetch_sales_data_from_db()
        
        # Check if result is None or empty list
        if result is None or len(result) == 0:
            print("Warning: Sales query returned empty results")
            return pd.DataFrame()
       
        # Convert to DataFrame
        data = []
        for row in result:
            data.append({
                'ds': row['order_date'],
                'y': float(row['total_sales'])
            })
       
        df = pd.DataFrame(data)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds"])
        print(f"Sales data loaded: {len(df)} rows")
        forecast_logger.info(f"Sales data loaded: {len(df)} rows, from {df['ds'].min()} to {df['ds'].max()}")
        return df
       
    except Exception as e:
        forecast_logger.error(f"Error fetching sales data: {e}")
        print(f"Error fetching sales data: {e}")
        return pd.DataFrame()

async def train_forecast_model_async(model_path: str = "sales_forecast.pkl"):
    """
    Train the forecast model asynchronously
    """
    try:
        
        # Get sales data asynchronously
        forecast_logger.info("Starting forecast model training")
        df = await get_sales_data_for_forecasting()
        
        # Proper DataFrame emptiness check
        if df.empty or len(df) == 0:
            forecast_logger.warning("No sales data found to train the model")
            return {
                "success": False,
                "message": "No sales data found to train the model"
            }

        # Run Prophet training in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Train model in executor
        def train_model():
            forecast_logger.debug("Training Prophet model")
            model = Prophet()
            model.fit(df)
            return model
        
        model = await loop.run_in_executor(None, train_model)
        
        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        return {
            "success": True,
            "message": "Forecast model trained successfully",
            "rows": len(df),
            "date_range": {
                "start": df["ds"].min().strftime("%Y-%m-%d"),
                "end": df["ds"].max().strftime("%Y-%m-%d")
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error training forecast model: {str(e)}"
        }

async def generate_forecast_async(periods: int = 30, model_path: str = "sales_forecast.pkl"):
    """
    Generate forecast asynchronously
    """
    try:
        # Load model asynchronously using thread pool
        loop = asyncio.get_event_loop()
        
        def load_model():
            with open(model_path, "rb") as f:
                return pickle.load(f)
        
        model = await loop.run_in_executor(None, load_model)
        
        # Run Prophet operations in thread pool
        def make_forecast():
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
            
            # Convert to proper format
            result_list = []
            for _, row in result.iterrows():
                result_list.append({
                    "ds": row["ds"].strftime("%Y-%m-%d"),
                    "yhat": float(row["yhat"]),
                    "yhat_lower": float(row["yhat_lower"]),
                    "yhat_upper": float(row["yhat_upper"])
                })
            return result_list
        
        forecast_data = await loop.run_in_executor(None, make_forecast)
        return forecast_data
        
    except FileNotFoundError:
        raise Exception("No trained model found. Train first via /train_forecast")
    except Exception as e:
        raise Exception(f"Forecast error: {str(e)}")

async def generate_forecast_summary_async(question: str, forecast_data: list) -> str:
    """
    Generate a natural language summary of the forecast results using LLM asynchronously
    """
    # Check if forecast_data is empty or None
    if not forecast_data or not isinstance(forecast_data, list) or len(forecast_data) == 0:
        return "No forecast data available to generate summary."
    
    # Extract key statistics from forecast data
    values = [item["yhat"] for item in forecast_data]
    dates = [item["ds"] for item in forecast_data]
    
    # Calculate basic statistics
    avg_value = sum(values) / len(values)
    max_value = max(values)
    min_value = min(values)
    max_date = dates[values.index(max_value)]
    min_date = dates[values.index(min_value)]
    
    # Calculate trend
    first_value = values[0]
    last_value = values[-1]
    trend_direction = "increasing" if last_value > first_value else "decreasing" if last_value < first_value else "stable"
    trend_percentage = abs((last_value - first_value) / first_value * 100) if first_value != 0 else 0
    
    # Create prompt for LLM
    system_prompt = """You are a helpful data analyst that explains forecast results in simple, natural language.
    Provide a clear summary that anyone can understand, focusing on:
    1. What the forecast shows overall
    2. Key trends and patterns
    3. Important highs and lows
    4. Practical implications
    Keep it concise but informative, and avoid technical jargon."""
    
    user_prompt = f"""Based on the user's question: "{question}"

Here are the sales forecast results for the next {len(forecast_data)} days:
- Average daily sales: ${avg_value:,.2f}
- Maximum sales: ${max_value:,.2f} (on {max_date})
- Minimum sales: ${min_value:,.2f} (on {min_date})
- Overall trend: {trend_direction} ({trend_percentage:.1f}% change from start to end)

Please provide a clear, natural language summary that explains what this forecast means in simple terms."""
    
    try:
        # Use async LLM completion
        summary = await llm_complete_async(system_prompt, user_prompt, temperature=0.3)
        return summary
    except Exception as e:
        # Fallback to simple summary if LLM fails
        print(f"LLM summary generation failed, using fallback: {e}")
        return generate_simple_forecast_summary(question, forecast_data)

def generate_simple_forecast_summary(question: str, forecast_data: list) -> str:
    """
    Generate a simple natural language summary without external LLM
    """
    if not forecast_data or not isinstance(forecast_data, list) or len(forecast_data) == 0:
        return "No forecast data available."
    
    values = [item["yhat"] for item in forecast_data]
    dates = [item["ds"] for item in forecast_data]
    
    # Calculate statistics
    avg_value = sum(values) / len(values)
    max_value = max(values)
    min_value = min(values)
    max_date = dates[values.index(max_value)]
    min_date = dates[values.index(min_value)]
    
    # Calculate trend
    first_value = values[0]
    last_value = values[-1]
    trend = "increasing" if last_value > first_value else "decreasing" if last_value < first_value else "stable"
    trend_percentage = abs((last_value - first_value) / first_value * 100) if first_value != 0 else 0
    
    summary = f"Based on your question about '{question}', the sales forecast shows:\n\n"
    summary += f"• The average daily sales over the next {len(forecast_data)} days will be ${avg_value:,.2f}\n"
    summary += f"• The forecast indicates a {trend} trend, with a {trend_percentage:.1f}% change from start to end\n"
    summary += f"• The highest predicted sales is ${max_value:,.2f} on {max_date}\n"
    summary += f"• The lowest predicted sales is ${min_value:,.2f} on {min_date}\n\n"
    summary += "These predictions include uncertainty ranges, meaning actual sales may vary within the predicted confidence intervals."
    
    return summary