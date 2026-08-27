# LangChain Learning

A hands-on repository for learning LangChain concepts from fundamentals to RAG and tools.

## Topics Covered

### LangChain Basics
- LangChain fundamentals
- Messages
- ChatPromptTemplate
- MessagesPlaceholder
- PromptTemplate

### Output Processing
- Output Parsers
- Structured Output
- Pydantic Output Parser

### Chains
- Sequential Chains
- Parallel Chains
- RunnableSequence
- RunnableParallel

### Document Processing
- Document Loaders
- Text Loader
- PDF Loader
- CSV Loader
- WebBaseLoader
- Directory Loader

### Text Splitting
- CharacterTextSplitter
- RecursiveCharacterTextSplitter
- Semantic Chunking
- Python Code Splitter

### Embeddings & Vector Stores
- Embeddings
- Vector Stores
- Chroma
- Retrievers

### RAG
- Basic RAG
- Retrieval
- RAG Systems

### Tools
- Tools
- StructuredTool
- Toolkits
- Tool Binding
- Tool Calling
- Tool Execution



# 🤖 Projects

The repository also contains hands-on projects that apply the concepts learned throughout the tutorials.

## Project 1 — Tool-Calling Agent

A dynamic **tool-calling agent** built with LangChain.

The agent allows an LLM to:

- Understand the user's request
- Decide whether a tool is required
- Select the appropriate tool
- Generate tool arguments
- Execute the selected tool dynamically
- Receive the tool result
- Send the result back to the LLM
- Generate a final response
- Handle multiple tool calls

### Project File

```text
Projects/
└── 01_Tool_Calling_Agent.ipynb
 ```


# 🏥 Hospital RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system built with LangChain for answering questions from hospital knowledge-base documents and synthetic patient records.

This project focuses on understanding and implementing the core RAG pipeline, from document ingestion and chunking to vector retrieval, hybrid retrieval, cross-encoder reranking, tool-based retrieval, and evaluation.

> Note: The hospital information and patient records used in this project are fictional/synthetic and are intended only for learning and RAG experimentation. This project is not connected to a real hospital and does not provide medical advice.


## 📌 Project Overview

The objective of this project is to build a hospital question-answering system that retrieves relevant information from different data sources and uses an LLM to generate grounded answers.

The system works with two primary information sources:

1. Hospital Knowledge Base
   - Hospital departments
   - Hospital services
   - Diagnostic services
   - Visiting hours
   - Pharmacy information
   - Other hospital-related information

2. Synthetic Patient Records
   - Patient ID
   - Name
   - Birth date
   - Gender
   - Marital status
   - Birthplace
   - City
   - State
   - ZIP code
   - Income
   - Healthcare expenses
   - Healthcare coverage

The system can also handle queries that require information from multiple sources.


# 🏗️ Architecture

                         User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │   Query Input   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Retrieval Layer │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
       Dense Retrieval               Sparse Retrieval
      Vector Similarity                    BM25
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                    Hybrid Retrieval
                             │
                             ▼
                    Top-5 Candidates
                             │
                             ▼
                 Cross-Encoder Reranker
                             │
                             ▼
                    Top-3 Candidates
                             │
                             ▼
                    Context Assembly
                             │
                             ▼
                           LLM
                             │
                             ▼
                      Final Answer


# 🔄 RAG Pipeline

The current RAG pipeline is:
``` text
Documents
    ↓
Document Loading
    ↓
Text Splitting
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
    ↓
Dense Retrieval
    ↓
BM25 Retrieval
    ↓
Hybrid Retrieval
    ↓
Top-5 Candidate Chunks
    ↓
Cross-Encoder Reranking
    ↓
Top-3 Relevant Chunks
    ↓
Context
    ↓
LLM
    ↓
Final Answer
```

# 📂 Document Ingestion

The project starts by loading hospital-related documents and preparing them for retrieval.

The ingestion process includes:

- Loading documents
- Extracting document content
- Splitting documents into chunks
- Generating embeddings
- Storing embeddings for retrieval

The quality of document processing directly affects downstream retrieval performance.


# ✂️ Chunking

Chunking is an important part of the RAG pipeline because documents are divided into smaller pieces before embeddings are generated.

The current chunking configuration is:

Chunk Size: 400
Chunk Overlap: 60

The 400/60 configuration was used for the current retrieval experiments.

Chunking helps the system:

- Preserve relevant context
- Avoid excessively large retrieval units
- Reduce irrelevant information
- Improve the quality of retrieved context

The project also explored the impact of chunking on retrieval quality.


# 🧠 Embeddings

Document chunks are converted into vector representations using an embedding model.

The basic process is:
``` text 
Document Chunk
      ↓
Embedding Model
      ↓
Vector Representation
      ↓
Vector Store

```

When a user asks a question, the query is also converted into a vector.

The system then compares the query vector with stored document vectors to identify semantically relevant chunks.


# 🔎 Vector Similarity Retrieval

The first retrieval approach implemented in the project was vector similarity retrieval.

The process is:
``` text
User Query
    ↓
Query Embedding
    ↓
Vector Similarity Search
    ↓
Relevant Document Chunks
```
Vector similarity retrieval is useful when the wording of the query and the document is different but the underlying meaning is similar.

Example:

Query:

"Which department treats children?"

Relevant document:

"The Pediatrics Department provides healthcare services for children and adolescents."

The wording is different, but semantic retrieval can identify the relationship.


# 🔀 Hybrid Retrieval

The project was extended from basic vector retrieval to Hybrid RAG.

Hybrid retrieval combines:

- Dense vector retrieval
- Sparse keyword retrieval using BM25

Architecture:

                    Query
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       Vector Search        BM25 Search
             │                 │
             └────────┬────────┘
                      │
                      ▼
              Hybrid Retrieval
                      │
                      ▼
              Candidate Chunks

Dense retrieval is useful for semantic matching.

BM25 is useful for exact or keyword-oriented matching, especially for:

- Specific terms
- Names
- IDs
- Department names
- Medical terminology
- Exact phrases

Combining both approaches provides a more robust retrieval strategy.


# 🎯 Retrieval and Reranking

The project uses a two-stage retrieval approach.

Stage 1: Candidate Retrieval

The retrieval layer initially retrieves:

Top K = 5

candidate chunks.

Stage 2: Cross-Encoder Reranking

The five retrieved candidates are then passed through a cross-encoder reranker.

The reranker selects the most relevant candidates.

Current configuration:

Initial Retrieval: Top 5
Reranker Output: Top 3

Pipeline:

``` text

Query
  ↓
Hybrid Retrieval
  ↓
Top 5 Candidates
  ↓
Cross-Encoder
  ↓
Relevance Scoring
  ↓
Top 3 Candidates
  ↓
LLM

```


# 🤖 Cross-Encoder Reranking

The project uses Sentence-Transformers for cross-encoder reranking.

Model used:

cross-encoder/ms-marco-MiniLM-L-6-v2

A cross-encoder evaluates the query and candidate document together.

Conceptually:

Query + Document 1 → Relevance Score
Query + Document 2 → Relevance Score
Query + Document 3 → Relevance Score
Query + Document 4 → Relevance Score
Query + Document 5 → Relevance Score

The candidates are then ranked based on their relevance scores.

This allows the system to retrieve a larger candidate set first and then use a more detailed relevance model to select the strongest context.


# 🛠️ Tool-Based Retrieval

The system uses LangChain tools to retrieve information from different sources.

## Hospital Search

Tool:

hospital_search

This tool is used for hospital knowledge-base questions such as:

- Hospital departments
- Hospital services
- Cardiology services
- Diagnostic services
- ICU visiting hours
- Pharmacy services
- Other hospital information

## Patient Search

Tool:

search_patient

This tool is used for synthetic patient-record queries.

Examples:

"What is the gender of patient <patient_id>?"

"What is the birth date of patient <patient_id>?"

"What city does patient <patient_id> live in?"

"What is the healthcare coverage of patient <patient_id>?"

"What is the complete information for patient <patient_id>?"


# 🔀 Mixed Queries

The system can handle queries that require information from multiple sources.

Example:

"What services are available in the Cardiology Department, and what is the gender of patient <patient_id>?"

This query requires:

hospital_search
+
search_patient

The system retrieves the relevant information from both sources and combines the results into a single answer.


# ❓ Direct Queries

The system was tested using direct hospital knowledge-base queries.

Examples:

"What are the ICU visiting hours?"

"What services are available in the Cardiology Department?"

"Which department provides care for children and adolescents?"

"What services does the hospital provide?"

"Does the hospital provide pharmacy services?"

"What diagnostic services are available at the hospital?"

"What departments are available at ABC General Hospital?"


# 🚫 Missing Information Queries

The system was also tested with queries where the requested information is not available in the knowledge base.

Example:

"What is the cardiologist's phone number?"

If the information is not present in the retrieved documents, the system should not fabricate an answer.

Expected behavior:

"I don't have that information in the current knowledge base."

This is important for reducing hallucination and keeping the generated answer grounded in the available context.


# 🧪 Evaluation and Testing

The current evaluation was performed using representative query categories rather than a large formal benchmark.

The following query types were tested:

## 1. Direct Queries

Questions that directly ask for information contained in the hospital documents.

Example:

"What are the ICU visiting hours?"


## 2. Semantic Queries

Questions that require semantic understanding rather than exact keyword matching.

Example:

"Which department provides care for children?"


## 3. Missing Information Queries

Questions where the requested information does not exist in the available knowledge base.

Example:

"What is the cardiologist's phone number?"


## 4. Patient Queries

Questions that require searching synthetic patient records.

Example:

"What is the birth date of patient <patient_id>?"


## 5. Mixed Queries

Questions that require multiple tools or information sources.

Example:

"What services are available in Cardiology, and what is the gender of patient <patient_id>?"


# 📊 Current Configuration

| Component | Configuration |
|---|---|
| Chunk Size | 400 |
| Chunk Overlap | 60 |
| Retrieval Strategy | Hybrid Retrieval |
| Dense Retrieval | Vector Similarity |
| Sparse Retrieval | BM25 |
| Initial Retrieval | Top 5 |
| Reranking | Cross-Encoder |
| Reranker Output | Top 3 |
| Reranker Model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Framework | LangChain |
| Data | Fictional Hospital + Synthetic Patient Records |


# 🧩 Technologies Used

- Python
- LangChain
- LangChain Core
- Vector Embeddings
- Vector Similarity Search
- BM25
- Hybrid Retrieval
- Sentence-Transformers
- Cross-Encoder Reranking
- Retrieval-Augmented Generation (RAG)
- LangChain Tools
- Synthetic Healthcare Data


# 🔬 Key Concepts Learned

## Document Processing

- Document loading
- Text splitting
- Chunk size
- Chunk overlap
- Context preservation

## Retrieval

- Embeddings
- Vector similarity
- Semantic retrieval
- BM25
- Hybrid retrieval

## Reranking

- Candidate retrieval
- Cross-encoder reranking
- Relevance scoring
- Top-K selection
- Improving retrieved context

## RAG Quality

- Direct query testing
- Semantic query testing
- Missing-information testing
- Patient-record testing
- Mixed-query testing
- Retrieval relevance
- Context quality
- Grounded answer generation

## Tool Calling

- LangChain tools
- Multiple retrieval tools
- Tool selection
- Multi-source retrieval
- Combining information from multiple tools


# 🎯 Current Project Status

The core RAG experimentation phase is complete.

The project has covered:

``` text 

Document Loading
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Similarity Retrieval
      ↓
Hybrid Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Tool-Based Retrieval
      ↓
Direct Queries
      ↓
Missing Information Queries
      ↓
Patient Queries
      ↓
Mixed Queries
      ↓
Testing

```

The current focus is to move forward rather than continue optimizing the RAG pipeline in depth.


# ⚠️ Disclaimer

This project uses fictional hospital information and synthetic patient records for educational and experimentation purposes.

It is not connected to a real hospital.

The information in this project must not be used for real-world medical decisions.


# 👨‍💻 Author

Mukkoteswara Rao Bodepudi

GenAI / AI Engineer

GitHub:
github.com/MukkoteswaraRao-Bodepudi