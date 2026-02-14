"""
Latency testing script for Customer Support Chatbot API
Tests various queries and reports latency statistics
"""

import requests
import time
import statistics
from typing import List, Dict
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

API_URL = "http://localhost:8000"

# Test queries covering all buckets
TEST_QUERIES = [
    # BUCKET_A (Direct responses - should be fastest)
    "How can I track my order?",
    "What are your delivery options?",
    "What payment methods do you accept?",
    "What is your refund policy?",
    
    # BUCKET_B (RAG + LLM - medium latency)
    "I need help with setting up my account",
    "How do I change my shipping address?",
    "Can you explain your warranty terms?",
    "What should I do if my item is damaged?",
    
    # BUCKET_C (Escalation - fast, no LLM)
    "I want to speak to a human representative",
    "This is unacceptable, I demand a refund!",
    "Connect me to your manager immediately",
]


def test_single_query(query: str, show_response: bool = False) -> Dict:
    """Test a single query and return timing information"""
    try:
        # Measure client-side latency
        start_time = time.time()
        
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": query},
            timeout=30
        )
        
        client_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        if response.status_code == 200:
            data = response.json()
            server_latency = data.get('latency_ms', 0)
            network_latency = client_latency - server_latency
            
            result = {
                "query": query,
                "success": True,
                "bucket": data.get('bucket', 'UNKNOWN'),
                "intent": data.get('intent', 'UNKNOWN'),
                "client_latency_ms": round(client_latency, 2),
                "server_latency_ms": round(server_latency, 2),
                "network_latency_ms": round(network_latency, 2),
                "response": data.get('response', '')
            }
            
            if show_response:
                print(f"\n{Fore.CYAN}Response: {data.get('response', '')[:100]}...")
            
            return result
        else:
            return {
                "query": query,
                "success": False,
                "error": f"HTTP {response.status_code}",
                "client_latency_ms": round(client_latency, 2)
            }
    
    except Exception as e:
        return {
            "query": query,
            "success": False,
            "error": str(e),
            "client_latency_ms": 0
        }


def print_result(result: Dict):
    """Print formatted result for a single query"""
    if result['success']:
        bucket = result['bucket']
        bucket_color = {
            'BUCKET_A': Fore.GREEN,
            'BUCKET_B': Fore.YELLOW,
            'BUCKET_C': Fore.RED
        }.get(bucket, Fore.WHITE)
        
        print(f"\n{Fore.CYAN}Query: {result['query']}")
        print(f"{bucket_color}Bucket: {bucket} | Intent: {result['intent']}")
        print(f"{Fore.WHITE}⏱️  Client: {result['client_latency_ms']}ms | "
              f"Server: {result['server_latency_ms']}ms | "
              f"Network: {result['network_latency_ms']}ms")
    else:
        print(f"\n{Fore.RED}❌ Query: {result['query']}")
        print(f"{Fore.RED}Error: {result['error']}")


def print_statistics(results: List[Dict]):
    """Print summary statistics"""
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print(f"\n{Fore.RED}❌ No successful requests!")
        return
    
    # Overall statistics
    client_latencies = [r['client_latency_ms'] for r in successful_results]
    server_latencies = [r['server_latency_ms'] for r in successful_results]
    
    # Bucket-wise statistics
    bucket_stats = {}
    for bucket in ['BUCKET_A', 'BUCKET_B', 'BUCKET_C']:
        bucket_results = [r for r in successful_results if r['bucket'] == bucket]
        if bucket_results:
            bucket_latencies = [r['server_latency_ms'] for r in bucket_results]
            bucket_stats[bucket] = {
                'count': len(bucket_results),
                'mean': statistics.mean(bucket_latencies),
                'median': statistics.median(bucket_latencies),
                'min': min(bucket_latencies),
                'max': max(bucket_latencies)
            }
    
    print(f"\n{'='*70}")
    print(f"{Fore.GREEN}📊 LATENCY STATISTICS")
    print(f"{'='*70}")
    
    print(f"\n{Fore.CYAN}Overall Performance:")
    print(f"  Total Requests: {len(results)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(results) - len(successful_results)}")
    
    print(f"\n{Fore.YELLOW}Client-side Latency (includes network + server):")
    print(f"  Mean:   {statistics.mean(client_latencies):.2f} ms")
    print(f"  Median: {statistics.median(client_latencies):.2f} ms")
    print(f"  Min:    {min(client_latencies):.2f} ms")
    print(f"  Max:    {max(client_latencies):.2f} ms")
    if len(client_latencies) > 1:
        print(f"  StdDev: {statistics.stdev(client_latencies):.2f} ms")
    
    print(f"\n{Fore.YELLOW}Server Processing Time:")
    print(f"  Mean:   {statistics.mean(server_latencies):.2f} ms")
    print(f"  Median: {statistics.median(server_latencies):.2f} ms")
    print(f"  Min:    {min(server_latencies):.2f} ms")
    print(f"  Max:    {max(server_latencies):.2f} ms")
    if len(server_latencies) > 1:
        print(f"  StdDev: {statistics.stdev(server_latencies):.2f} ms")
    
    # Bucket-wise breakdown
    print(f"\n{Fore.MAGENTA}Bucket Performance Breakdown:")
    for bucket, stats in bucket_stats.items():
        bucket_color = {
            'BUCKET_A': Fore.GREEN,
            'BUCKET_B': Fore.YELLOW,
            'BUCKET_C': Fore.RED
        }[bucket]
        
        print(f"\n{bucket_color}{bucket} ({stats['count']} requests):")
        print(f"  Mean:   {stats['mean']:.2f} ms")
        print(f"  Median: {stats['median']:.2f} ms")
        print(f"  Range:  {stats['min']:.2f} - {stats['max']:.2f} ms")
    
    # Performance insights
    print(f"\n{Fore.CYAN}💡 Performance Insights:")
    
    if 'BUCKET_A' in bucket_stats and 'BUCKET_B' in bucket_stats:
        speedup = bucket_stats['BUCKET_B']['mean'] / bucket_stats['BUCKET_A']['mean']
        print(f"  • BUCKET_A is {speedup:.1f}x faster than BUCKET_B (template vs RAG+LLM)")
    
    if 'BUCKET_A' in bucket_stats:
        a_mean = bucket_stats['BUCKET_A']['mean']
        if a_mean < 100:
            print(f"  • BUCKET_A latency is excellent (<100ms) ✓")
        elif a_mean < 200:
            print(f"  • BUCKET_A latency is good (<200ms) ✓")
        else:
            print(f"  • BUCKET_A latency could be improved (>{a_mean:.0f}ms)")
    
    if 'BUCKET_B' in bucket_stats:
        b_mean = bucket_stats['BUCKET_B']['mean']
        if b_mean < 1000:
            print(f"  • BUCKET_B latency is good (<1s) ✓")
        elif b_mean < 2000:
            print(f"  • BUCKET_B latency is acceptable (<2s) ✓")
        else:
            print(f"  • BUCKET_B latency needs optimization (>{b_mean/1000:.1f}s)")
    
    network_overhead = statistics.mean([r['network_latency_ms'] for r in successful_results])
    print(f"  • Average network overhead: {network_overhead:.2f} ms")


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"{Fore.GREEN}✓ API is healthy")
            return True
        else:
            print(f"{Fore.RED}✗ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"{Fore.RED}✗ Cannot connect to API: {e}")
        print(f"{Fore.YELLOW}Make sure the API is running on {API_URL}")
        return False


def main():
    """Run latency tests"""
    print(f"{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}🚀 Customer Support Chatbot - Latency Test")
    print(f"{Fore.CYAN}{'='*70}\n")
    
    # Check API health
    if not check_api_health():
        return
    
    print(f"\n{Fore.YELLOW}Running {len(TEST_QUERIES)} test queries...\n")
    
    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"{Fore.WHITE}[{i}/{len(TEST_QUERIES)}]", end=" ")
        result = test_single_query(query, show_response=False)
        print_result(result)
        results.append(result)
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Print statistics
    print_statistics(results)
    
    print(f"\n{Fore.GREEN}{'='*70}")
    print(f"{Fore.GREEN}✓ Latency test complete!")
    print(f"{Fore.GREEN}{'='*70}\n")


if __name__ == "__main__":
    main()
