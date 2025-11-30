import os
import argparse
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

from rag_engine import RAGEngine
from llm_client import LLMClient
from telemetry import TelemetryLogger

load_dotenv()

app = Flask(__name__)

# Initialize components
rag = RAGEngine()
llm = LLMClient(
    host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
    model=os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
)
logger = TelemetryLogger(log_dir=os.getenv('LOG_DIR', './logs'))

# Configuration
MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', 10000))
TOP_K_CHUNKS = int(os.getenv('TOP_K_CHUNKS', 3))

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Quiz Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; color: white; margin-bottom: 40px; }
        header h1 { font-size: 3em; margin-bottom: 10px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: white; border-radius: 12px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        textarea { width: 100%; height: 300px; padding: 15px; border: 2px solid #e0e0e0; 
                   border-radius: 8px; font-size: 14px; resize: vertical; }
        textarea:focus { outline: none; border-color: #667eea; }
        input, select { width: 100%; padding: 12px; border: 2px solid #e0e0e0; 
                        border-radius: 8px; font-size: 14px; margin-bottom: 10px; }
        button { width: 100%; padding: 15px; background: #667eea; color: white; 
                 border: none; border-radius: 8px; font-size: 16px; font-weight: bold; 
                 cursor: pointer; transition: all 0.3s; }
        button:hover { background: #5568d3; transform: translateY(-2px); }
        button:disabled { background: #ccc; cursor: not-allowed; transform: none; }
        .quiz-output { min-height: 300px; }
        .question { border-left: 4px solid #667eea; padding-left: 20px; margin-bottom: 30px; }
        .question h3 { color: #333; margin-bottom: 15px; }
        .hint, .rubric { padding: 12px; border-radius: 6px; margin-top: 10px; font-size: 14px; }
        .hint { background: #fff3cd; border-left: 3px solid #ffc107; }
        .rubric { background: #d1ecf1; border-left: 3px solid #17a2b8; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; 
                 border-left: 4px solid #f5c6cb; margin-bottom: 20px; }
        .loading { text-align: center; padding: 60px; color: #666; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; 
                   border-radius: 50%; width: 50px; height: 50px; 
                   animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; 
                 margin-top: 20px; font-size: 12px; }
        .stat { background: #f8f9fa; padding: 10px; border-radius: 6px; text-align: center; }
        .stat-value { font-size: 20px; font-weight: bold; color: #667eea; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📚 Quiz Generator</h1>
            <p>Transform your study notes into educational quizzes with AI</p>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2 style="margin-bottom: 20px;">Study Notes</h2>
                <textarea id="notes" placeholder="Paste your study notes here...

Example:
Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in chloroplasts and requires sunlight, water, and carbon dioxide..."></textarea>
                
                <div style="margin-top: 20px;">
                    <input type="text" id="topic" placeholder="Specific topic (optional)">
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <select id="difficulty">
                            <option value="easy">Easy</option>
                            <option value="medium" selected>Medium</option>
                            <option value="hard">Hard</option>
                        </select>
                        
                        <select id="num">
                            <option value="1">1 Question</option>
                            <option value="3" selected>3 Questions</option>
                            <option value="5">5 Questions</option>
                        </select>
                    </div>
                    
                    <button onclick="generate()" id="genBtn">Generate Quiz</button>
                </div>
                
                <div class="stats" id="stats" style="display: none;">
                    <div class="stat">
                        <div class="stat-value" id="latency">-</div>
                        <div>Latency</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="tokens">-</div>
                        <div>Tokens</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="chunks">-</div>
                        <div>Chunks</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2 style="margin-bottom: 20px;">Generated Quiz</h2>
                <div id="output" class="quiz-output">
                    <div style="text-align: center; padding: 80px 20px; color: #999;">
                        <div style="font-size: 48px; margin-bottom: 20px;">📝</div>
                        <p>Your quiz will appear here</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function generate() {
            const notes = document.getElementById('notes').value;
            const topic = document.getElementById('topic').value;
            const difficulty = document.getElementById('difficulty').value;
            const num = document.getElementById('num').value;
            const btn = document.getElementById('genBtn');
            const output = document.getElementById('output');
            const stats = document.getElementById('stats');
            
            if (!notes.trim()) {
                output.innerHTML = '<div class="error">Please provide study notes.</div>';
                return;
            }
            
            btn.disabled = true;
            btn.textContent = 'Generating...';
            stats.style.display = 'none';
            output.innerHTML = '<div class="loading"><div class="spinner"></div><p>Retrieving context and generating questions...</p></div>';
            
            try {
                const start = Date.now();
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ notes, topic, difficulty, num_questions: parseInt(num) })
                });
                
                const data = await res.json();
                
                if (!res.ok) {
                    output.innerHTML = `<div class="error">${data.error || 'Generation failed'}</div>`;
                    return;
                }
                
                // Display quiz
                let html = '';
                data.questions.forEach((q, i) => {
                    html += `
                        <div class="question">
                            <h3>Question ${i+1}: ${q.question}</h3>
                            <div class="hint"><strong>💡 Hint:</strong> ${q.hint}</div>
                            <div class="rubric"><strong>📋 Rubric:</strong> ${q.rubric}</div>
                        </div>
                    `;
                });
                output.innerHTML = html;
                
                // Show stats
                stats.style.display = 'grid';
                document.getElementById('latency').textContent = `${Date.now() - start}ms`;
                document.getElementById('tokens').textContent = data.telemetry?.tokens || '~500';
                document.getElementById('chunks').textContent = data.telemetry?.chunks || TOP_K_CHUNKS;
                
            } catch (err) {
                output.innerHTML = `<div class="error">Network error: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Quiz';
            }
        }
        
        // Allow Enter to submit (with Ctrl/Cmd)
        document.getElementById('notes').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                generate();
            }
        });
    </script>
</body>
</html>
"""


def validate_input(notes, topic):
    """Validate and sanitize inputs"""
    if not notes or not notes.strip():
        return False, "Notes cannot be empty"
    
    if len(notes) > MAX_INPUT_LENGTH:
        return False, f"Notes too long. Maximum {MAX_INPUT_LENGTH} characters."
    
    # Check for prompt injection patterns
    injection_patterns = [
        'ignore previous instructions',
        'disregard above',
        'forget everything',
        'you are now',
        'new instructions:',
        'ignore all previous',
    ]
    
    combined = (notes + ' ' + topic).lower()
    for pattern in injection_patterns:
        if pattern in combined:
            return False, "Invalid input detected. Please rephrase your request."
    
    return True, None


@app.route('/')
def index():
    """Serve web interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate_quiz():
    """Main quiz generation endpoint"""
    import time
    start_time = time.time()
    
    try:
        data = request.json
        notes = data.get('notes', '')
        topic = data.get('topic', '')
        difficulty = data.get('difficulty', 'medium')
        num_questions = int(data.get('num_questions', 3))
        
        # Validate input
        valid, error = validate_input(notes, topic)
        if not valid:
            logger.log('RAG', 0, 'error', error=error)
            return jsonify({'error': error}), 400
        
        # Build corpus and retrieve context
        chunks = rag.chunk_text(notes)
        rag.build_index(chunks)
        
        query = topic if topic else "generate quiz questions from notes"
        context_chunks = rag.retrieve(query, k=TOP_K_CHUNKS)
        context = "\n\n".join(context_chunks)
        
        # Generate quiz with LLM
        quiz = llm.generate_quiz(
            context=context,
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        # Calculate metrics
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log telemetry
        logger.log(
            pathway='RAG',
            latency_ms=latency_ms,
            status='success',
            tokens_input=len(context.split()) * 1.3,  # Rough estimate
            tokens_output=len(str(quiz).split()) * 1.3,
            chunks_retrieved=len(context_chunks)
        )
        
        return jsonify({
            'questions': quiz,
            'telemetry': {
                'latency_ms': latency_ms,
                'chunks': len(context_chunks),
                'tokens': int(len(context.split()) * 1.3 + len(str(quiz).split()) * 1.3)
            }
        })
        
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.log('RAG', latency_ms, 'error', error=str(e))
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


def cli_mode(args):
    """CLI interface"""
    print("🎓 Quiz Generator CLI\n")
    
    # Load notes
    if args.notes.endswith('.md') or args.notes.endswith('.txt'):
        with open(args.notes, 'r') as f:
            notes = f.read()
    else:
        notes = args.notes
    
    print(f"Notes length: {len(notes)} characters")
    print(f"Topic: {args.topic or 'Auto-detected'}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Questions: {args.num}\n")
    
    # Validate
    valid, error = validate_input(notes, args.topic)
    if not valid:
        print(f"❌ Error: {error}")
        return
    
    # Generate
    print("⏳ Generating quiz...\n")
    import time
    start = time.time()
    
    chunks = rag.chunk_text(notes)
    rag.build_index(chunks)
    context_chunks = rag.retrieve(args.topic or "quiz questions", k=TOP_K_CHUNKS)
    context = "\n\n".join(context_chunks)
    
    quiz = llm.generate_quiz(context, args.topic, args.difficulty, args.num)
    
    elapsed = time.time() - start
    
    # Display
    for i, q in enumerate(quiz, 1):
        print(f"{'='*60}")
        print(f"Question {i}: {q['question']}\n")
        print(f"💡 Hint: {q['hint']}\n")
        print(f"📋 Rubric: {q['rubric']}\n")
    
    print(f"{'='*60}")
    print(f"\n⏱️  Generated in {elapsed:.2f}s")
    print(f"📊 Retrieved {len(context_chunks)} context chunks")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quiz Generator')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    parser.add_argument('--notes', type=str, help='Path to notes file or text')
    parser.add_argument('--topic', type=str, default='', help='Specific topic')
    parser.add_argument('--difficulty', type=str, default='medium', 
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--num', type=int, default=3, help='Number of questions')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    
    args = parser.parse_args()
    
    if args.cli:
        if not args.notes:
            print("Error: --notes required in CLI mode")
            exit(1)
        cli_mode(args)
    else:
        print("🚀 Starting Quiz Generator web server...")
        print(f"📡 Open http://localhost:{args.port} in your browser")
        app.run(host='0.0.0.0', port=args.port, debug=False)