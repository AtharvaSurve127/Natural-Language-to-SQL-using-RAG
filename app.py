from flask import Flask, render_template, request, jsonify, session
import json
import time
import os
import mysql.connector
from mysql.connector import Error
import chromadb
from pymongo import MongoClient
import google.generativeai as genai
from datetime import date, datetime
import re
import markdown
from bs4 import BeautifulSoup
import sqlparse
from functools import wraps
import logging
from decimal import Decimal
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-production')

# Configuration
class Config:
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', 'chats')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDRHGpp0ZUhaJNoGXXmz23l-KLE5HtHlt4')
    MYSQL_CONFIG = {
        'database': os.environ.get('MYSQL_DATABASE', 'parks_and_recreation'),
        'user': os.environ.get('MYSQL_USER', 'root'),
        'password': os.environ.get('MYSQL_PASSWORD', 'Cr7rocks'),
        'host': os.environ.get('MYSQL_HOST', 'localhost')
    }
    ALLOWED_OPERATIONS = ['SELECT', 'INSERT', 'DELETE', 'UPDATE']
    PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'C:\\Users\\Admin\\OneDrive\\Desktop\\Text2SQL\\Data')

app.config.from_object(Config)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom JSON Encoder to handle Decimal types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super(CustomJSONEncoder, self).default(obj)

app.json_encoder = CustomJSONEncoder

# Initialize clients
try:
    mongo_client = MongoClient(app.config['MONGO_URI'])
    chat_db = mongo_client[app.config['MONGO_DB_NAME']]
    history_collection = chat_db['history2']
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    mongo_client = None

try:
    genai.configure(api_key=app.config['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("Gemini AI model configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini AI: {e}")
    model = None

# Helper Functions
def get_schema_query(db_name):
    return f"""
    SELECT TABLE_NAME as table_name, 
           COLUMN_NAME as column_name, 
           DATA_TYPE as data_type, 
           IS_NULLABLE as is_nullable, 
           COLUMN_KEY as column_key,
           COLUMN_DEFAULT as column_default,
           EXTRA as extra
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{db_name}'
    ORDER BY TABLE_NAME, ORDINAL_POSITION;
    """

def get_relationship_query():
    return """
    SELECT 
        TABLE_SCHEMA as table_schema,
        TABLE_NAME as table_name,
        COLUMN_NAME as column_name,
        REFERENCED_TABLE_SCHEMA as foreign_table_schema,
        REFERENCED_TABLE_NAME as foreign_table_name,
        REFERENCED_COLUMN_NAME as foreign_column_name
    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE REFERENCED_TABLE_NAME IS NOT NULL
        AND TABLE_SCHEMA = DATABASE();
    """

def get_mysql_connection(database_name=None):
    """Create and return a MySQL connection to the specified database (or default)."""
    try:
        config = app.config['MYSQL_CONFIG'].copy()
        if database_name:
            config['database'] = database_name
        connection = mysql.connector.connect(**config)
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        raise

def execute_query_with_result(query, database_name=None):
    """Execute a query and return results (for SELECT queries)"""
    try:
        connection = get_mysql_connection(database_name)
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        # Format datetime and Decimal objects for JSON serialization
        formatted_result = []
        for row in result:
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, (datetime, date)):
                    formatted_row[key] = value.isoformat()
                elif isinstance(value, Decimal):
                    formatted_row[key] = float(value)
                else:
                    formatted_row[key] = value
            formatted_result.append(formatted_row)
        return formatted_result
    except Error as e:
        logger.error(f"Error executing query: {e}")
        return {'error': str(e)}

def execute_modification_query(query, database_name=None):
    """Execute INSERT, UPDATE, or DELETE queries and return affected rows"""
    try:
        connection = get_mysql_connection(database_name)
        cursor = connection.cursor()
        cursor.execute(query)
        affected_rows = cursor.rowcount
        connection.commit()
        cursor.close()
        connection.close()
        return {"success": True, "affected_rows": affected_rows}
    except Error as e:
        logger.error(f"Error executing modification query: {e}")
        return {'error': str(e)}

def store_embeddings(data, collection_name, chroma_client):
    if not data:
        logger.warning(f"No data to store in collection: {collection_name}")
        return

    # Convert data to text format
    texts = []
    metadata = []
    
    for row in data:
        # Create a descriptive text string for embedding
        text_parts = []
        meta = {}
        for key, value in row.items():
            if value is not None:
                text_parts.append(f"{key}: {value}")
                meta[key] = str(value)
        texts.append(" ".join(text_parts))
        metadata.append(meta)

    # Ensure we have valid data
    if not texts:
        logger.warning(f"No valid texts generated for collection: {collection_name}")
        return

    try:
        # Get or create collection
        try:
            collection = chroma_client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            collection = chroma_client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")

        # Generate unique IDs
        ids = [f"{collection_name}_{i}" for i in range(len(texts))]

        # Add documents to collection
        collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadata
        )
        logger.info(f"Successfully stored {len(texts)} documents in collection: {collection_name}")

    except Exception as e:
        logger.error(f"Error storing embeddings in collection {collection_name}: {e}")
        raise

def clean_sql_query(text):
    try:
        # Convert markdown to HTML
        html = markdown.markdown(text)
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # If there's a code block, get its content
        code_block = soup.find('code')
        if code_block:
            text = code_block.get_text()
        
        # Remove any remaining backticks
        text = text.replace('`', '')
        
        # Remove 'sql' keyword if present at the start
        text = re.sub(r'^sql\s+', '', text, flags=re.IGNORECASE)
        
        # Format the SQL query using sqlparse
        text = sqlparse.format(
            text,
            keyword_case='UPPER',
            identifier_case='lower',
            reindent=True,
            strip_comments=True
        )
        
        # Ensure query ends with semicolon
        if not text.rstrip().endswith(';'):
            text = text.rstrip() + ';'
        
        # Remove any extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    except Exception as e:
        logger.error(f"Error cleaning SQL query: {e}")
        # If parsing fails, return the original text with basic cleaning
        text = text.replace('`', '').strip()
        if not text.endswith(';'):
            text += ';'
        return text

def extract_sql_operation_type(query):
    """Extract the operation type (SELECT, INSERT, UPDATE, DELETE) from a SQL query"""
    try:
        # Remove leading comments and whitespace
        query = re.sub(r"^--.*\n", "", query, flags=re.MULTILINE)
        query = query.strip().upper()
        for operation in app.config['ALLOWED_OPERATIONS']:
            if query.startswith(operation):
                return operation
        return None
    except Exception as e:
        logger.error(f"Error extracting SQL operation type: {e}")
        return None

def execute_generated_query(query, database_name=None):
    """Execute the generated SQL query based on its operation type"""
    operation_type = extract_sql_operation_type(query)
    if not operation_type:
        return {'error': 'Unsupported SQL operation'}
    if operation_type == 'SELECT':
        return execute_query_with_result(query, database_name)
    elif operation_type in ['INSERT', 'UPDATE', 'DELETE']:
        return execute_modification_query(query, database_name)
    else:
        return {'error': f'Operation {operation_type} not supported'}

def generate_natural_language_response(user_query, sql_query, execution_result, operation_type):
    """
    Generate a natural language response from SQL query results based on operation type
    """
    try:
        # If there's an error in the result
        if isinstance(execution_result, dict) and 'error' in execution_result:
            return f"I encountered an error while executing the query: {execution_result['error']}"
        
        # Handle different operation types
        if operation_type == 'SELECT':
            # If no results were returned
            if not execution_result:
                return "I couldn't find any data matching your query."
            
            # For average salary query specifically
            if re.search(r'average.*salary', user_query, re.IGNORECASE) or "AVG" in sql_query and "salary" in sql_query.lower():
                if len(execution_result) == 1:
                    # Extract the average salary value
                    avg_salary = None
                    for key, value in execution_result[0].items():
                        if isinstance(value, (float, int)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                            avg_salary = value
                            break
                    
                    if avg_salary is not None:
                        # Format as currency
                        if isinstance(avg_salary, (float, int)):
                            formatted_salary = f"${avg_salary:,.2f}"
                        else:
                            try:
                                formatted_salary = f"${float(avg_salary):,.2f}"
                            except:
                                formatted_salary = avg_salary
                        
                        return f"The average salary of employees is {formatted_salary}."
            
            # Generic response for other types of SELECT queries
            if len(execution_result) == 1:
                # For single row results
                result_str = ", ".join([f"{k}: {v}" for k, v in execution_result[0].items()])
                return f"I found the following result: {result_str}"
            else:
                # For multiple row results, show a summary of all fields for each result (up to 10)
                max_results = 10
                shown_results = execution_result[:max_results]
                lines = [f"{i+1}. " + ", ".join([f"{k}: {v}" for k, v in row.items()]) for i, row in enumerate(shown_results)]
                summary = "\n".join(lines)
                more_count = len(execution_result) - max_results
                if more_count > 0:
                    summary += f"\n...and {more_count} more results."
                return f"I found {len(execution_result)} results matching your query:\n{summary}"
        
        # Handle INSERT operations
        elif operation_type == 'INSERT':
            affected_rows = execution_result.get('affected_rows', 0)
            if affected_rows == 1:
                return f"Successfully inserted 1 row into the database."
            else:
                return f"Successfully inserted {affected_rows} rows into the database."
        
        # Handle DELETE operations
        elif operation_type == 'DELETE':
            affected_rows = execution_result.get('affected_rows', 0)
            if affected_rows == 0:
                return "No rows were deleted. The specified record may not exist."
            elif affected_rows == 1:
                return "Successfully deleted 1 row from the database."
            else:
                return f"Successfully deleted {affected_rows} rows from the database."
        
        # Handle UPDATE operations
        elif operation_type == 'UPDATE':
            affected_rows = execution_result.get('affected_rows', 0)
            if affected_rows == 0:
                return "No rows were updated. The specified record may not exist or the new values are the same as the old ones."
            elif affected_rows == 1:
                return "Successfully updated 1 row in the database."
            else:
                return f"Successfully updated {affected_rows} rows in the database."
        
    except Exception as e:
        logger.error(f"Error generating natural language response: {e}")
        return "I processed your query, but encountered an issue formatting the response."
    
    # Fallback response with raw data
    return "Here are the results of your query."

# Routes
@app.route('/')
def chatbot_home():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    if request.method == 'POST':
        data = request.get_json()
        user_query = data.get('message')
        project_name = data.get('project_name', 'default_project')
        natural_language = data.get('natural_language', True)
        operation_type = data.get('operation_type', 'any')
        database_name = data.get('database_name', app.config['MYSQL_CONFIG']['database'])
        if not user_query:
            return jsonify({'error': 'No message provided.'}), 400
        try:
            persist_directory = app.config['PERSIST_DIRECTORY']
            os.makedirs(persist_directory, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=persist_directory)
            collections = {}
            for name in ["schema_embeddings_MySQL", "relationship_embeddings_MySQL"]:
                try:
                    collections[name] = chroma_client.get_collection(name=name)
                except Exception:
                    collections[name] = chroma_client.create_collection(name=name)
            if any(len(collection.get()['ids']) == 0 for collection in collections.values()):
                connection = get_mysql_connection(database_name)
                cursor = connection.cursor(dictionary=True)
                cursor.execute(get_schema_query(database_name))
                schema_result = cursor.fetchall()
                cursor.execute(get_relationship_query())
                relationship_result = cursor.fetchall()
                cursor.close()
                connection.close()
                store_embeddings(schema_result, "schema_embeddings_MySQL", chroma_client)
                store_embeddings(relationship_result, "relationship_embeddings_MySQL", chroma_client)
            schema_results = collections["schema_embeddings_MySQL"].query(
                query_texts=[user_query],
                n_results=10
            )
            relationship_results = collections["relationship_embeddings_MySQL"].query(
                query_texts=[user_query],
                n_results=10
            )
            schema_context = "\n".join([doc for doc in schema_results['documents'][0]]) if schema_results['documents'] else ""
            relationship_context = "\n".join([doc for doc in relationship_results['documents'][0]]) if relationship_results['documents'] else ""
            if operation_type.lower() == 'select':
                operation_instruction = "Generate only a valid SQL SELECT query."
            elif operation_type.lower() == 'insert':
                operation_instruction = "Generate only a valid SQL INSERT query."
            elif operation_type.lower() == 'delete':
                operation_instruction = "Generate only a valid SQL DELETE query."
            elif operation_type.lower() == 'update':
                operation_instruction = "Generate only a valid SQL UPDATE query."
            else:
                operation_instruction = "Generate a valid SQL query (SELECT, INSERT, DELETE, or UPDATE)."
            context = f"""
            You are an expert MySQL database developer. Below is the schema and 
            relationship information for the database. Based on this information, 
            {operation_instruction} that addresses the following user query.
            
            Schema information:
            {schema_context}

            Relationship information:
            {relationship_context}

            User query:
            {user_query}

            Important: Format the SQL query as a markdown code block with sql language specified.
            Make sure the query is syntactically valid and follows MySQL best practices.
            For INSERT, UPDATE, and DELETE queries, be extremely careful to include proper WHERE clauses
            to prevent unintended data modifications.
            """
            response = model.generate_content(context)
            ai_response = clean_sql_query(response.text)
            logger.info(f"Cleaned SQL for operation extraction: {ai_response}")
            detected_operation = extract_sql_operation_type(ai_response)
            execution_result = execute_generated_query(ai_response, database_name)
            nl_response = None
            if natural_language:
                nl_response = generate_natural_language_response(user_query, ai_response, execution_result, detected_operation)
            if mongo_client:
                history_entry = {
                    'user_query': user_query,
                    'ai_response': ai_response,
                    'operation_type': detected_operation,
                    'execution_result': execution_result,
                    'natural_language_response': nl_response,
                    'timestamp': time.time(),
                    'project_name': project_name,
                    'model_used': 'gemini',
                    'database_name': database_name
                }
                history_collection.insert_one(history_entry)
            return jsonify({
                'raw_result': execution_result,
                'sql_query': ai_response,
                'operation_type': detected_operation,
                'response': nl_response if natural_language else execution_result
            })
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/add_data_source', methods=['POST'])
def add_data_source():
    if request.method == 'POST':
        data = request.get_json()
        project_name = data.get('project_name', 'default_project')
        
        if not project_name:
            return jsonify({'error': 'No project name provided.'}), 400

        try:
            connection = get_mysql_connection()
            cursor = connection.cursor(dictionary=True)

            cursor.execute(get_schema_query(app.config['MYSQL_CONFIG']['database']))
            schema_result = cursor.fetchall()

            cursor.execute(get_relationship_query())
            relationship_result = cursor.fetchall()

            cursor.close()
            connection.close()

            persist_directory = app.config['PERSIST_DIRECTORY']
            os.makedirs(persist_directory, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=persist_directory)
            
            store_embeddings(schema_result, "schema_embeddings_MySQL", chroma_client)
            store_embeddings(relationship_result, "relationship_embeddings_MySQL", chroma_client)

            return jsonify({'success': True, 'message': 'Data source added and embeddings stored successfully.'})

        except Exception as e:
            logger.error(f"Failed to add data source: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/execute_custom_sql', methods=['POST'])
def execute_custom_sql():
    """Endpoint to execute custom SQL queries with validation"""
    if request.method == 'POST':
        data = request.get_json()
        sql_query = data.get('sql_query')
        project_name = data.get('project_name', 'default_project')
        database_name = data.get('database_name', app.config['MYSQL_CONFIG']['database'])
        if not sql_query:
            return jsonify({'error': 'No SQL query provided.'}), 400
        try:
            clean_query = clean_sql_query(sql_query)
            operation_type = extract_sql_operation_type(clean_query)
            if not operation_type:
                return jsonify({'error': 'Unsupported SQL operation'}), 400
            if operation_type not in app.config['ALLOWED_OPERATIONS']:
                return jsonify({'error': f'Operation {operation_type} not allowed'}), 403
            result = execute_generated_query(clean_query, database_name)
            if mongo_client:
                history_entry = {
                    'user_query': "Custom SQL execution",
                    'ai_response': clean_query,
                    'operation_type': operation_type,
                    'execution_result': result,
                    'timestamp': time.time(),
                    'project_name': project_name,
                    'source': 'custom_sql',
                    'database_name': database_name
                }
                history_collection.insert_one(history_entry)
            return jsonify({
                'raw_result': result,
                'sql_query': clean_query,
                'operation_type': operation_type
            })
        except Exception as e:
            logger.error(f"Failed to execute custom SQL: {e}")
            return jsonify({'error': str(e)}), 500

# Utility endpoint to get table structures
@app.route('/get_tables', methods=['GET'])
def get_tables():
    """Get all tables in the database"""
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Query to get all tables
        cursor.execute(f"SHOW TABLES FROM {app.config['MYSQL_CONFIG']['database']};")
        tables = [list(table.values())[0] for table in cursor.fetchall()]
        
        cursor.close()
        connection.close()
        
        return jsonify({'tables': tables})
    
    except Exception as e:
        logger.error(f"Failed to get tables: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_table_structure', methods=['GET'])
def get_table_structure():
    """Get structure of a specific table"""
    table_name = request.args.get('table_name')
    
    if not table_name:
        return jsonify({'error': 'No table name provided'}), 400
        
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get table structure
        cursor.execute(f"DESCRIBE {table_name};")
        structure = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return jsonify({'structure': structure})
    
    except Exception as e:
        logger.error(f"Failed to get table structure: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_er_diagram', methods=['POST'])
def get_er_diagram():
    data = request.get_json()
    database_name = data.get('database_name', app.config['MYSQL_CONFIG']['database'])
    connection = get_mysql_connection(database_name)
    cursor = connection.cursor(dictionary=True)
    # Get tables and columns
    cursor.execute(get_schema_query(database_name))
    schema = cursor.fetchall()
    # Get relationships
    cursor.execute(get_relationship_query())
    relationships = cursor.fetchall()
    cursor.close()
    connection.close()

    # Build Mermaid ER diagram definition
    tables = {}
    for row in schema:
        table = row['table_name']
        col = row['column_name']
        dtype = row['data_type']
        if table not in tables:
            tables[table] = []
        tables[table].append(f"{col} {dtype}")

    mermaid = ["erDiagram"]
    for table, cols in tables.items():
        mermaid.append(f"    {table} {{")
        for col in cols:
            mermaid.append(f"        {col}")
        mermaid.append("    }")
    for rel in relationships:
        mermaid.append(f"    {rel['table_name']} ||--o{{ {rel['foreign_table_name']} : \"FK\"")
    mermaid_code = "\n".join(mermaid)
    return jsonify({"mermaid": mermaid_code})

if __name__ == '__main__':
    app.run(debug=True)