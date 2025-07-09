#!/usr/bin/env python3
"""
Profiling script for KnowAI to identify bottlenecks in the workflow.

Usage:
    python scripts/profile_knowai.py [vectorstore_path]

This script profiles the KnowAI workflow using a specific question and configuration
to identify performance bottlenecks in each node.
"""

import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai.core import KnowAIAgent


@dataclass
class NodeProfile:
    """Profile data for a single node."""
    node_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: str = ""


class KnowAIProfiler:
    """Profiler for KnowAI workflow performance."""
    
    def __init__(self, vectorstore_path: str):
        self.vectorstore_path = vectorstore_path
        self.profiles: List[NodeProfile] = []
        self.agent = None
        
    async def profile_workflow(self, question: str, selected_files: List[str], detailed_response: bool = True):
        """Profile the complete KnowAI workflow."""
        print(f"🔍 Profiling KnowAI workflow...")
        print(f"   Question: {question}")
        print(f"   Files: {selected_files}")
        print(f"   Detailed Response: {detailed_response}")
        print(f"   Vectorstore: {self.vectorstore_path}")
        print("-" * 80)
        
        # Initialize agent with profiling
        start_time = time.perf_counter()
        try:
            self.agent = KnowAIAgent(
                vectorstore_path=self.vectorstore_path,
                log_graph=True  # Enable graph logging to see the workflow
            )
            init_time = time.perf_counter() - start_time
            print(f"✅ Agent initialization: {init_time:.4f}s")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            return
        
        # Profile the main workflow
        workflow_start = time.perf_counter()
        try:
            result = await self.agent.process_turn(
                user_question=question,
                selected_files=selected_files,
                detailed_response_desired=detailed_response
            )
            workflow_time = time.perf_counter() - workflow_start
            
            print(f"\n📊 Workflow Results:")
            print(f"   Total workflow time: {workflow_time:.4f}s")
            print(f"   Generation length: {len(result.get('generation', ''))} characters")
            print(f"   Files processed: {len(result.get('documents_by_file', {}))}")
            
            # Show a preview of the response
            generation = result.get('generation', '')
            if generation:
                preview = generation[:200] + "..." if len(generation) > 200 else generation
                print(f"\n📝 Response Preview:")
                print(f"   {preview}")
            
        except Exception as e:
            print(f"❌ Workflow execution failed: {e}")
            return
    
    def analyze_performance(self):
        """Analyze the collected performance data."""
        if not self.profiles:
            print("No performance data collected.")
            return
        
        print(f"\n📈 Performance Analysis:")
        print("-" * 80)
        
        # Sort by duration
        sorted_profiles = sorted(self.profiles, key=lambda x: x.duration, reverse=True)
        
        total_time = sum(p.duration for p in self.profiles)
        
        print(f"{'Node':<30} {'Duration (s)':<15} {'% of Total':<12} {'Status':<10}")
        print("-" * 80)
        
        for profile in sorted_profiles:
            percentage = (profile.duration / total_time * 100) if total_time > 0 else 0
            status = "✅" if profile.success else "❌"
            print(f"{profile.node_name:<30} {profile.duration:<15.4f} {percentage:<12.1f}% {status:<10}")
        
        print("-" * 80)
        print(f"{'TOTAL':<30} {total_time:<15.4f} {'100.0':<12}%")
        
        # Identify bottlenecks
        if sorted_profiles:
            bottleneck = sorted_profiles[0]
            print(f"\n🚨 Primary Bottleneck: {bottleneck.node_name} ({bottleneck.duration:.4f}s)")
            
            if len(sorted_profiles) > 1:
                second_bottleneck = sorted_profiles[1]
                print(f"🔶 Secondary Bottleneck: {second_bottleneck.node_name} ({second_bottleneck.duration:.4f}s)")


async def main():
    """Main profiling function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/profile_knowai.py <vectorstore_path>")
        print("Example: python scripts/profile_knowai.py /path/to/vectorstore")
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
    
    # Create profiler and run
    profiler = KnowAIProfiler(vectorstore_path)
    
    print(f"🚀 Starting KnowAI Profiling Session")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        await profiler.profile_workflow(question, selected_files, detailed_response)
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