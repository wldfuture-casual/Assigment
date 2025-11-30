import os
import json
from datetime import datetime
from typing import Optional


class TelemetryLogger:
    """Logger for tracking LLM requests and performance"""
    
    def __init__(self, log_dir='./logs'):
        """Initialize logger with output directory"""
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'telemetry.jsonl')
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"✓ Telemetry logging to: {self.log_file}")
    
    def log(
        self,
        pathway: str,
        latency_ms: int,
        status: str,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        chunks_retrieved: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a request
        
        Args:
            pathway: Processing pathway (e.g., 'RAG', 'tool', 'direct')
            latency_ms: Request latency in milliseconds
            status: 'success' or 'error'
            tokens_input: Number of input tokens (optional)
            tokens_output: Number of output tokens (optional)
            chunks_retrieved: Number of RAG chunks retrieved (optional)
            error: Error message if status is 'error' (optional)
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'pathway': pathway,
            'latency_ms': latency_ms,
            'status': status,
        }
        
        if tokens_input is not None:
            log_entry['tokens_input'] = int(tokens_input)
        
        if tokens_output is not None:
            log_entry['tokens_output'] = int(tokens_output)
        
        if tokens_input and tokens_output:
            log_entry['tokens_total'] = int(tokens_input + tokens_output)
        
        if chunks_retrieved is not None:
            log_entry['chunks_retrieved'] = chunks_retrieved
        
        if error:
            log_entry['error'] = str(error)
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_stats(self) -> dict:
        """Get aggregated statistics from logs"""
        if not os.path.exists(self.log_file):
            return {
                'total_requests': 0,
                'success_rate': 0,
                'avg_latency_ms': 0,
                'total_tokens': 0
            }
        
        total = 0
        successes = 0
        latencies = []
        tokens = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    total += 1
                    
                    if entry.get('status') == 'success':
                        successes += 1
                    
                    if 'latency_ms' in entry:
                        latencies.append(entry['latency_ms'])
                    
                    if 'tokens_total' in entry:
                        tokens.append(entry['tokens_total'])
                        
                except json.JSONDecodeError:
                    continue
        
        return {
            'total_requests': total,
            'success_rate': (successes / total * 100) if total > 0 else 0,
            'avg_latency_ms': sum(latencies) // len(latencies) if latencies else 0,
            'total_tokens': sum(tokens),
            'avg_tokens_per_request': sum(tokens) // len(tokens) if tokens else 0
        }
    
    def print_stats(self) -> None:
        """Print formatted statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("TELEMETRY STATISTICS")
        print("="*60)
        print(f"Total Requests:       {stats['total_requests']}")
        print(f"Success Rate:         {stats['success_rate']:.1f}%")
        print(f"Avg Latency:          {stats['avg_latency_ms']}ms")
        print(f"Total Tokens:         {stats['total_tokens']}")
        print(f"Avg Tokens/Request:   {stats['avg_tokens_per_request']}")
        print("="*60 + "\n")
    
    def tail_logs(self, n: int = 10) -> None:
        """Print last n log entries"""
        if not os.path.exists(self.log_file):
            print("No logs yet.")
            return
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        print(f"\n{'='*80}")
        print(f"LAST {min(n, len(lines))} LOG ENTRIES")
        print('='*80)
        
        for line in lines[-n:]:
            try:
                entry = json.loads(line)
                timestamp = entry['timestamp'][:19].replace('T', ' ')
                pathway = entry['pathway'].ljust(10)
                status = entry['status'].ljust(7)
                latency = f"{entry['latency_ms']}ms".ljust(8)
                
                line_str = f"{timestamp} | {pathway} | {status} | {latency}"
                
                if 'tokens_total' in entry:
                    line_str += f" | {entry['tokens_total']} tokens"
                
                if entry['status'] == 'error' and 'error' in entry:
                    line_str += f" | ERROR: {entry['error'][:40]}"
                
                print(line_str)
                
            except json.JSONDecodeError:
                continue
        
        print('='*80 + "\n")


if __name__ == '__main__':
    # Test telemetry logger
    logger = TelemetryLogger('./test_logs')
    
    # Simulate some requests
    print("Logging sample requests...")
    
    logger.log('RAG', 1250, 'success', tokens_input=450, tokens_output=320, chunks_retrieved=3)
    logger.log('RAG', 980, 'success', tokens_input=380, tokens_output=290, chunks_retrieved=3)
    logger.log('RAG', 1500, 'error', error='Connection timeout')
    logger.log('RAG', 1100, 'success', tokens_input=420, tokens_output=310, chunks_retrieved=2)
    
    # Print stats
    logger.print_stats()
    
    # Show recent logs
    logger.tail_logs(5)