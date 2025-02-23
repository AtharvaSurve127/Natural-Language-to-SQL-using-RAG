# Natural Language to SQL using RAG

A RAG-powered system that converts natural language questions into accurate SQL queries. Leveraging ChromaDB for semantic retrieval and Gemini Pro for generation, the system ensures precise SQL queries by augmenting prompts with actual database schema and relationships.

## Features

- Natural language to SQL conversion using RAG architecture
- Automatic schema and relationship understanding
- Real-time query execution
- Query history tracking
- SQL injection prevention
- Web-based chat interface

## RAG Architecture

The system implements Retrieval-Augmented Generation (RAG) for accurate SQL generation:

1. **Retrieval**: 
   - ChromaDB stores embeddings of database schema and relationships
   - Semantic search retrieves relevant table structures and connections
   
2. **Augmentation**:
   - Retrieved context is formatted into structured prompts
   - Includes exact table names, columns, and relationships
   
3. **Generation**:
   - Gemini Pro generates SQL using augmented context
   - Ensures accurate table/column names and proper JOINs
   - Prevents hallucination of non-existent database structures

## How It Works

1. **Query Processing**
   - User enters natural language question
   - System retrieves relevant database context
   - Gemini Pro generates optimized SQL query
   - Query is validated and executed
   - Results are returned to user

2. **Schema Understanding**
   - ChromaDB stores database schema embeddings
   - Semantic search finds relevant tables
   - Relationship mapping ensures proper JOINs

## Technology Stack

- **Backend**: Flask (Python)
- **Databases**: 
  - MySQL (Main database)
  - MongoDB (Query history)
  - ChromaDB (Vector store for RAG)
- **LLM**: Google Gemini Pro
- **Frontend**: HTML/CSS/JavaScript

## Quick Start

1. Set up environment:
```bash
pip install -r requirements.txt
```

2. Configure credentials in `.env`:
```
GEMINI_API_KEY=your_key
MYSQL_CONFIG=your_config
MONGO_URI=your_uri
```

3. Run the application:
```bash
python app.py
```

4. Visit `http://localhost:5000` in your browser

## Example Usage

Enter questions like:
- "Show all employees in Parks department"
- "What is Leslie's current position?"
- "List projects with their teams"

## License

MIT License
