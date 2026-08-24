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