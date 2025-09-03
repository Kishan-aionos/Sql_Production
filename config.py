import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD", "AVNS_MWRv5iMXRum9DMQ6cPV")
DB_NAME = os.getenv("DB_NAME", "defaultdb")
DB_CA_CERT = os.getenv("DB_CA_CERT")

# LLM configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in environment variables.")
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"


# App configuration
MODEL_PATH = "sales_forecast.pkl"

# Schema hint for LLM
SCHEMA_HINT = """
Tables:
- customers(customer_id, company_name, contact_name, country, region)
- categories(category_id, category_name)
- products(product_id, product_name, category_id, unit_price)
- orders(order_id, customer_id, order_date, ship_country, ship_region)
- order_details(order_id, product_id, unit_price, quantity, discount)

Important:
- the table order detail is incorrect the correct table is order_details.
- To calculate sales use: order_details.unit_price * order_details.quantity * (1 - IFNULL(order_details.discount,0))
- Join orders with order_details using orders.order_id = order_details.order_id
- Join products with order_details using products.product_id = order_details.product_id
- Join categories with products using categories.category_id = products.category_id
- Use exact table names as defined in schema (customers, categories, products, orders, order_details).
- Do NOT invent or modify table names (e.g., no spaces, always use underscores).
"""