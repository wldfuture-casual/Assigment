# Quiz Generator - RAG-Powered Learning Tool

An LLM application that transforms study notes into educational quizzes with hints and grading rubrics using RAG (Retrieval Augmented Generation).

## Features

- **RAG Enhancement**: Embeds notes into chunks, uses vector similarity search to retrieve relevant context
- **Safety**: System prompts with explicit rules, prompt injection detection, input length guards
- **Telemetry**: Logs timestamp, pathway, latency, and token counts per request
- **Offline Eval**: 15+ test cases with automated pass rate reporting
- **Robust Error Handling**: Graceful fallbacks for all failure modes

## Architecture

```
User Input (Notes + Topic)
    ↓
Input Validation (Length, Injection)
    ↓
Chunk Notes → Generate Embeddings
    ↓
RAG Retrieval (Top-K similar chunks)
    ↓
LLM Call (Ollama) with System Prompt + Context
    ↓
Quiz Generation (Questions + Hints + Rubrics)
    ↓
Telemetry Logging
```

## Prerequisites

- Python 3.8+
- Ollama installed and running
- 4GB+ RAM recommended

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd quiz-generator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama (separate terminal)
ollama serve

# 4. Pull required model
ollama pull llama3.2:3b

# 5. Run the application
python app.py

# 6. Run offline tests
python test_eval.py
```

## Project Structure

```
quiz-generator/
├── app.py                 # Main Flask application
├── rag_engine.py          # RAG implementation (embedding + retrieval)
├── llm_client.py          # Ollama client with safety checks
├── telemetry.py           # Request logging
├── test_eval.py           # Offline evaluation script
├── tests.json             # Test cases (15+ inputs)
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── data/                  # Sample study notes
│   ├── biology.md
│   ├── physics.md
│   └── programming.py
└── logs/                  # Telemetry logs (auto-created)
```

## Usage

### Web Interface (CLI commands also available)

```bash
# Start server
python app.py

# Navigate to http://localhost:5000
```

### CLI Mode

```bash
python app.py --cli --notes data/biology.md --topic "Photosynthesis" --num 3 --difficulty medium
```

### API Endpoint

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "notes": "Your study notes here...",
    "topic": "Specific topic",
    "difficulty": "medium",
    "num_questions": 3
  }'
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# App settings
MAX_INPUT_LENGTH=10000
TOP_K_CHUNKS=3
EMBEDDING_MODEL=all-minilm

# Telemetry
LOG_DIR=./logs
```

## Safety Features

1. **System Prompt**: Explicit rules preventing harmful outputs
2. **Input Validation**: 
   - Length limit (10,000 chars)
   - Prompt injection patterns blocked
3. **Error Handling**: Graceful fallbacks with user-friendly messages
4. **Sanitization**: All outputs sanitized before display

## RAG Implementation

- **Chunking**: Splits notes into semantic paragraphs (min 50 chars)
- **Embedding**: Uses sentence-transformers (all-MiniLM-L6-v2)
- **Retrieval**: Cosine similarity search, top-K chunks
- **Context Window**: Fits retrieved chunks within token limits

## Telemetry

Logs stored in `logs/telemetry.jsonl`:

```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "pathway": "RAG",
  "latency_ms": 1250,
  "tokens_input": 450,
  "tokens_output": 320,
  "status": "success",
  "model": "llama3.2:3b"
}
```

## Offline Evaluation

Run tests:
```bash
python test_eval.py
```

Expected output:
```
Running 15 test cases...
✓ Test 1: Valid notes → Success
✓ Test 2: Injection attempt → Blocked
✓ Test 3: Oversized input → Error handled
...
Pass Rate: 15/15 (100%)
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linter
flake8 .

# Format code
black .

# Type check
mypy app.py
```

## Troubleshooting

**Ollama not responding:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

**Model not found:**
```bash
# List available models
ollama list

# Pull required model
ollama pull llama3.2:3b
```

**Import errors:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

