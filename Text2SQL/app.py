
from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)

# Configuration
class Config:
    MONGO_URI = 'mongodb://localhost:27017/'
    MONGO_DB_NAME = 'chats'
    GEMINI_API_KEY = 'AIzaSyANZ67Z_psb8oYbQg1R0lv4trNsryeuN6M'
    MYSQL_CONFIG = {
        'database': 'parks_and_recreation',
        'user': 'root',
        'password': 'Cr7rocks',
        'host': 'localhost'
    }

app.config.from_object(Config)

# Initialize clients
mongo_client = MongoClient(app.config['MONGO_URI'])
chat_db = mongo_client[app.config['MONGO_DB_NAME']]
history_collection = chat_db['history2']

genai.configure(api_key=app.config['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-pro')

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
            keyword_case='upper',
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
        print(f"Error cleaning SQL query: {e}")
        # If parsing fails, return the original text with basic cleaning
        text = text.replace('`', '').strip()
        if not text.endswith(';'):
            text += ';'
        return text

def execute_generated_query(query):
    try:
        connection = mysql.connector.connect(**app.config['MYSQL_CONFIG'])
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()

        # Format datetime objects
        formatted_result = []
        for row in result:
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, (datetime, date)):
                    formatted_row[key] = value.isoformat()
                else:
                    formatted_row[key] = value
            formatted_result.append(formatted_row)
        
        return formatted_result

    except Error as e:
        print(f"Error executing query: {e}")
        return {'error': str(e)}

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

def store_embeddings(data, collection_name, chroma_client):
    if not data:
        print(f"No data to store in collection: {collection_name}")
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
        print(f"No valid texts generated for collection: {collection_name}")
        return

    try:
        # Get or create collection
        try:
            collection = chroma_client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except Exception:
            collection = chroma_client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")

        # Generate unique IDs
        ids = [f"{collection_name}_{i}" for i in range(len(texts))]

        # Add documents to collection
        collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadata
        )
        print(f"Successfully stored {len(texts)} documents in collection: {collection_name}")

    except Exception as e:
        print(f"Error storing embeddings in collection {collection_name}: {e}")
        raise

@app.route('/')
def chatbot_home():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    if request.method == 'POST':
        data = request.get_json()
        user_query = data.get('message')
        project_name = data.get('project_name', 'default_project')

        if not user_query:
            return jsonify({'error': 'No message provided.'}), 400

        try:
            persist_directory = r'C:\Users\Admin\OneDrive\Desktop\Text2SQL\Data'
            print(f"Persist directory: {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=persist_directory)

            
            collections = {}
            for name in ["schema_embeddings_MySQL", "relationship_embeddings_MySQL"]:
                try:
                    collections[name] = chroma_client.get_collection(name=name)
                except Exception:
                    collections[name] = chroma_client.create_collection(name=name)

            # Check if collections are empty and populate if needed
            if any(len(collection.get()['ids']) == 0 for collection in collections.values()):
                connection = mysql.connector.connect(**app.config['MYSQL_CONFIG'])
                cursor = connection.cursor(dictionary=True)

                # Fetch schema and relationship data
                cursor.execute(get_schema_query(app.config['MYSQL_CONFIG']['database']))
                schema_result = cursor.fetchall()

                cursor.execute(get_relationship_query())
                relationship_result = cursor.fetchall()

                cursor.close()
                connection.close()

                # Store embeddings
                store_embeddings(schema_result, "schema_embeddings_MySQL", chroma_client)
                store_embeddings(relationship_result, "relationship_embeddings_MySQL", chroma_client)

            # Query collections
            schema_results = collections["schema_embeddings_MySQL"].query(
                query_texts=[user_query],
                n_results=10
            )
            relationship_results = collections["relationship_embeddings_MySQL"].query(
                query_texts=[user_query],
                n_results=10
            )

            # Prepare context
            schema_context = "\n".join([doc for doc in schema_results['documents'][0]]) if schema_results['documents'] else ""
            relationship_context = "\n".join([doc for doc in relationship_results['documents'][0]]) if relationship_results['documents'] else ""

            context = f"""
            You are an expert MySQL database developer. Below is the schema and 
            relationship information for the database. Based on this information, 
            generate an optimized and correct SQL SELECT query that answers the following user query.
            
            Schema information:
            {schema_context}

            Relationship information:
            {relationship_context}

            User query:
            {user_query}

            Important: Generate only a valid SQL query. Format it as a markdown code block with sql language specified.
            Example format:
            sql
            SELECT column FROM table WHERE condition;
            
            """

            response = model.generate_content(context)
            ai_response = clean_sql_query(response.text)
            print(f"Generated SQL Query: {ai_response}")

            execution_result = execute_generated_query(ai_response)

            # Store in history
            history_entry = {
                'user_query': user_query,
                'ai_response': ai_response,
                'execution_result': execution_result,
                'timestamp': time.time(),
                'project_name': project_name,
                'model_used': 'gemini',
            }
            history_collection.insert_one(history_entry)

            return jsonify({'response': execution_result})

        except Exception as e:
            print(f"Exception occurred: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/add_data_source', methods=['POST'])
def add_data_source():
    if request.method == 'POST':
        data = request.get_json()
        project_name = data.get('project_name', 'default_project')
        
        if not project_name:
            return jsonify({'error': 'No project name provided.'}), 400

        try:
            connection = mysql.connector.connect(**app.config['MYSQL_CONFIG'])
            cursor = connection.cursor(dictionary=True)

            # Fetch schema and relationship data
            cursor.execute(get_schema_query(app.config['MYSQL_CONFIG']['database']))
            schema_result = cursor.fetchall()

            cursor.execute(get_relationship_query())
            relationship_result = cursor.fetchall()

            cursor.close()
            connection.close()

            # Store embeddings
            persist_directory = r'C:\Users\Admin\OneDrive\Desktop\Text2SQL\Data'
            os.makedirs(persist_directory, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=persist_directory)
            
            store_embeddings(schema_result, "schema_embeddings_MySQL", chroma_client)
            store_embeddings(relationship_result, "relationship_embeddings_MySQL", chroma_client)

            return jsonify({'success': True, 'message': 'Data source added and embeddings stored successfully.'})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


