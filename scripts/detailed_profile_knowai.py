#!/usr/bin/env python3
"""
Detailed profiling script for KnowAI to identify bottlenecks in individual nodes.

Usage:
    python scripts/detailed_profile_knowai.py [vectorstore_path]

This script provides detailed profiling of each node in the KnowAI workflow
using a custom progress callback to capture timing information.
"""

import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai.core import KnowAIAgent


@dataclass
class NodeTiming:
    """Timing data for a single node execution."""
    node_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DetailedProfiler:
    """Detailed profiler for KnowAI workflow with node-level timing."""
    
    def __init__(self):
        self.node_timings: List[NodeTiming] = []
        self.current_node_start: float = 0
        self.current_node_name: str = ""
        
    def progress_callback(self, node_name: str, status: str, metadata: Dict[str, Any]):
        """Callback function to capture node progress and timing."""
        current_time = time.perf_counter()
        
        if status == "start":
            # Node is starting
            self.current_node_start = current_time
            self.current_node_name = node_name
            print(f"🔄 Starting: {node_name}")
            
        elif status == "end":
            # Node is ending
            if self.current_node_name == node_name:
                duration = current_time - self.current_node_start
                timing = NodeTiming(
                    node_name=node_name,
                    start_time=self.current_node_start,
                    end_time=current_time,
                    duration=duration,
                    success=True,
                    metadata=metadata
                )
                self.node_timings.append(timing)
                print(f"✅ Completed: {node_name} ({duration:.4f}s)")
                
        elif status == "error":
            # Node encountered an error
            if self.current_node_name == node_name:
                duration = current_time - self.current_node_start
                timing = NodeTiming(
                    node_name=node_name,
                    start_time=self.current_node_start,
                    end_time=current_time,
                    duration=duration,
                    success=False,
                    error_message=metadata.get("error", "Unknown error"),
                    metadata=metadata
                )
                self.node_timings.append(timing)
                print(f"❌ Failed: {node_name} ({duration:.4f}s) - {timing.error_message}")
    
    def analyze_performance(self):
        """Analyze the collected performance data."""
        if not self.node_timings:
            print("No timing data collected.")
            return
        
        print(f"\n📈 Detailed Performance Analysis:")
        print("=" * 80)
        
        # Sort by duration
        sorted_timings = sorted(self.node_timings, key=lambda x: x.duration, reverse=True)
        
        total_time = sum(t.duration for t in self.node_timings)
        
        print(f"{'Node':<35} {'Duration (s)':<15} {'% of Total':<12} {'Status':<10} {'Metadata':<20}")
        print("-" * 80)
        
        for timing in sorted_timings:
            percentage = (timing.duration / total_time * 100) if total_time > 0 else 0
            status = "✅" if timing.success else "❌"
            
            # Extract key metadata
            metadata_str = ""
            if timing.metadata:
                if "documents_retrieved" in timing.metadata:
                    metadata_str = f"docs: {timing.metadata['documents_retrieved']}"
                elif "queries_generated" in timing.metadata:
                    metadata_str = f"queries: {timing.metadata['queries_generated']}"
                elif "files_processed" in timing.metadata:
                    metadata_str = f"files: {timing.metadata['files_processed']}"
            
            print(f"{timing.node_name:<35} {timing.duration:<15.4f} {percentage:<12.1f}% {status:<10} {metadata_str:<20}")
        
        print("-" * 80)
        print(f"{'TOTAL':<35} {total_time:<15.4f} {'100.0':<12}%")
        
        # Identify bottlenecks
        if sorted_timings:
            bottleneck = sorted_timings[0]
            print(f"\n🚨 Primary Bottleneck: {bottleneck.node_name} ({bottleneck.duration:.4f}s, {bottleneck.duration/total_time*100:.1f}%)")
            
            if len(sorted_timings) > 1:
                second_bottleneck = sorted_timings[1]
                print(f"🔶 Secondary Bottleneck: {second_bottleneck.node_name} ({second_bottleneck.duration:.4f}s, {second_bottleneck.duration/total_time*100:.1f}%)")
        
        # Performance insights
        print(f"\n💡 Performance Insights:")
        
        # Check for LLM-heavy operations
        llm_nodes = [t for t in self.node_timings if any(keyword in t.node_name.lower() for keyword in ['llm', 'generate', 'combine'])]
        if llm_nodes:
            llm_total = sum(t.duration for t in llm_nodes)
            print(f"   • LLM operations: {llm_total:.4f}s ({llm_total/total_time*100:.1f}% of total)")
        
        # Check for retrieval operations
        retrieval_nodes = [t for t in self.node_timings if any(keyword in t.node_name.lower() for keyword in ['retrieve', 'extract', 'embed'])]
        if retrieval_nodes:
            retrieval_total = sum(t.duration for t in retrieval_nodes)
            print(f"   • Retrieval operations: {retrieval_total:.4f}s ({retrieval_total/total_time*100:.1f}% of total)")
        
        # Check for initialization operations
        init_nodes = [t for t in self.node_timings if any(keyword in t.node_name.lower() for keyword in ['instantiate', 'load'])]
        if init_nodes:
            init_total = sum(t.duration for t in init_nodes)
            print(f"   • Initialization operations: {init_total:.4f}s ({init_total/total_time*100:.1f}% of total)")


async def main():
    """Main profiling function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/detailed_profile_knowai.py <vectorstore_path>")
        print("Example: python scripts/detailed_profile_knowai.py /path/to/vectorstore")
        sys.exit(1)
    
    vectorstore_path = sys.argv[1]
    
    # Test configuration
    question = "List the vegetation management strategies in table format with citations"
    selected_files = ["Arizona_Public_Service_2024.pdf", "BC_Hydro_2020.pdf"]
    detailed_response = False
    
    # Configure logging for profiling
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create profiler
    profiler = DetailedProfiler()
    
    print(f"🚀 Starting Detailed KnowAI Profiling Session")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"🔍 Question: {question}")
    print(f"📁 Files: {selected_files}")
    print(f"⚡ Detailed Response: {detailed_response}")
    print(f"🗄️  Vectorstore: {vectorstore_path}")
    print("=" * 80)
    
    try:
        # Initialize agent
        print("🔧 Initializing KnowAI Agent...")
        agent = KnowAIAgent(
            vectorstore_path=vectorstore_path,
            log_graph=True
        )
        print("✅ Agent initialized successfully")
        
        # Profile the workflow
        print(f"\n🔄 Executing workflow...")
        workflow_start = time.perf_counter()
        
        result = await agent.process_turn(
            user_question=question,
            selected_files=selected_files,
            detailed_response_desired=detailed_response,
            progress_cb=profiler.progress_callback
        )
        
        workflow_time = time.perf_counter() - workflow_start
        
        print(f"\n📊 Workflow Results:")
        print(f"   Total workflow time: {workflow_time:.4f}s")
        print(f"   Generation length: {len(result.get('generation', ''))} characters")
        print(f"   Files processed: {len(result.get('documents_by_file', {}))}")
        
        # Show a preview of the response
        generation = result.get('generation', '')
        if generation:
            preview = generation[:300] + "..." if len(generation) > 300 else generation
            print(f"\n📝 Response Preview:")
            print(f"   {preview}")
        
        # Analyze performance
        profiler.analyze_performance()
        
    except KeyboardInterrupt:
        print("\n⏹️  Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Profiling session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main()) 