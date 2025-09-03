import re
import aiomysql
from typing import Any, Dict, List, Optional
from decimal import Decimal
from datetime import date, datetime
import os
import ssl
import asyncio

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_CA_CERT

# Allowed tables for queries
ALLOWED_TABLES = {"orders", "order_details", "products", "categories", "customers"}
READ_ONLY_PATTERNS = [r"^\s*select\b", r"^\s*with\b", r"^\s*show\b", r"^\s*describe\b", r"^\s*explain\b"]

async def get_connection_async():
    ssl_ctx = None
    if DB_CA_CERT and os.path.exists(DB_CA_CERT):
        try:
            # Create SSL context for aiomysql
            ssl_ctx = ssl.create_default_context(cafile=DB_CA_CERT)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED
        except Exception as e:
            print(f"SSL context creation failed: {e}")
            ssl_ctx = None
    elif DB_CA_CERT:
        print(f"Warning: SSL certificate file not found at {DB_CA_CERT}. Connecting without SSL.")

    connection_params = {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "db": DB_NAME,
        "autocommit": True,
        "charset": "utf8mb4",
    }

    if ssl_ctx:
        connection_params["ssl"] = ssl_ctx

    try:
        connection = await aiomysql.connect(**connection_params)
        return connection
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        raise Exception(f"Failed to connect to database: {str(e)}")

def _json_sanitize(val):
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (date, datetime)):
        return val.isoformat()
    return val

def is_read_only_sql(sql: str) -> bool:
    if not sql:
        return False
    s = sql.strip().lower()
    if any(re.match(p, s) for p in READ_ONLY_PATTERNS):
        forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"]
        return not any(f in s for f in forbidden)
    return False

def only_allowed_tables(sql: str) -> bool:
    if not sql:
        return False
    s = re.sub(r"`|\"", "", sql.lower())
    if re.search(r"\b(from|join)\b", s) and not any(re.search(rf"\b{t}\b", s) for t in ALLOWED_TABLES):
        return False
    suspects = re.findall(r"\bfrom\s+([a-zA-Z0-9_\.]+)", s) + re.findall(r"\bjoin\s+([a-zA-Z0-9_\.]+)", s)
    for name in suspects:
        short = name.split(".")[-1]
        if short not in ALLOWED_TABLES:
            return False
    return True

async def run_sql_async(sql: str) -> Dict[str, Any]:
    if not sql or not sql.strip():
        return {"message": "No SQL query provided"}
    
    if not is_read_only_sql(sql):
        return {"message": "This operation will not be performed (non-read-only query detected)."}
    
    if not only_allowed_tables(sql):
        return {"message": "Query references invalid or disallowed tables."}

    connection = None
    try:
        connection = await get_connection_async()
        async with connection.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql)
            if cur.description:
                columns = [col[0] for col in cur.description]
            else:
                columns = []
            rows = await cur.fetchall()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        raise Exception(f"Database error: {str(e)}")
    finally:
        if connection:
            connection.close()

async def get_sales_data_async():
    """
    Fetch daily total sales from orders + order_details for forecasting.
    """
    query = """
        SELECT
            o.order_date,
            SUM(od.unit_price * od.quantity * (1 - IFNULL(od.discount, 0))) AS total_sales
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        GROUP BY o.order_date
        ORDER BY o.order_date;
    """
    
    connection = None
    try:
        connection = await get_connection_async()
        async with connection.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query)
            rows = await cur.fetchall()
           
            if not rows:
                print("Warning: Sales query returned empty results")
                return None
           
            return rows
    except Exception as e:
        print(f"Error fetching sales data: {e}")
        return None
    finally:
        if connection:
            connection.close()

async def get_table_stats_async():
    """
    Get statistics about database tables for debugging
    """
    tables = ["orders", "order_details"]
    stats = {}
    
    connection = None
    try:
        connection = await get_connection_async()
        async with connection.cursor(aiomysql.DictCursor) as cur:
            for table in tables:
                await cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = await cur.fetchone()
                stats[table] = result["count"] if result else 0
        return stats
    except Exception as e:
        print(f"Error getting table stats: {e}")
        return {}
    finally:
        if connection:
            connection.close()

# For backward compatibility with existing code
async def get_connection():
    return await get_connection_async()

async def run_sql(sql: str):
    return await run_sql_async(sql)

async def get_sales_data():
    return await get_sales_data_async()