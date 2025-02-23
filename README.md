# Natural-Language-to-SQL-using-RAG
Convert natural language queries into SQL using Google's Gemini Pro API and semantic search. Get instant results from your MySQL database through an intuitive chat interface.
Features

Natural language to SQL conversion
Automatic schema and relationship understanding
Real-time query execution
Query history tracking
SQL injection prevention
Web-based chat interface

RAG Architecture
This project implements Retrieval-Augmented Generation (RAG) for accurate SQL generation:

Retrieval:

ChromaDB stores embeddings of database schema and relationships
Semantic search retrieves relevant table structures and connections


Augmentation:

Retrieved context is formatted into structured prompts
Includes exact table names, columns, and relationships


Generation:

Gemini Pro generates SQL using augmented context
Ensures accurate table/column names and proper JOINs
Prevents hallucination of non-existent database structures



How It Works

Query Processing

User enters natural language question
System retrieves relevant database context
Gemini Pro generates optimized SQL query
Query is validated and executed
Results are returned to user


Schema Understanding

ChromaDB stores database schema embeddings
Semantic search finds relevant tables
Relationship mapping ensures proper JOINs



Technology Stack

Backend: Flask (Python)
Databases:

MySQL (Main database)
MongoDB (Query history)
ChromaDB (Vector store for RAG)


LLM: Google Gemini Pro
Frontend: HTML/CSS/JavaScript

Quick Start

Set up environment:

bashCopypip install -r requirements.txt

Configure credentials in .env:

CopyGEMINI_API_KEY=your_key
MYSQL_CONFIG=your_config
MONGO_URI=your_uri

Run the application:

bashCopypython app.py

Visit http://localhost:5000 in your browser

Example Usage
Enter questions like:

"Show all employees in Parks department"
"What is Leslie's current position?"
"List projects with their teams"

License
MIT License
