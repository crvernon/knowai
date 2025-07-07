# KnowAI Profiling Guide

This guide explains how to profile KnowAI to identify performance bottlenecks and optimize the workflow.

## Overview

KnowAI is a conversational RAG agent built on LangGraph. Profiling helps identify which nodes in the workflow are taking the most time, allowing you to optimize performance.

## Quick Start

### Basic Profiling

```bash
# Run basic profiling with your vectorstore
python scripts/profile_knowai.py /path/to/your/vectorstore

# Example with specific files
python scripts/profile_knowai.py /path/to/vectorstore
```

### Detailed Profiling

```bash
# Run detailed profiling with node-level timing
python scripts/detailed_profile_knowai.py /path/to/your/vectorstore
```

## Test Configuration

The profiling scripts use this test configuration:

- **Question**: "List the vegetation management strategies in table format with citations"
- **Files**: ["Arizona_Public_Service_2024.pdf", "BC_Hydro_2020.pdf"]
- **Detailed Response**: False (for faster processing)

## Understanding the Workflow

KnowAI follows this workflow:

1. **Initialization Nodes**:
   - `instantiate_embeddings_node` - Loads Azure OpenAI embeddings
   - `instantiate_llm_large_node` - Loads large LLM for synthesis
   - `instantiate_llm_small_node` - Loads small LLM for query generation
   - `load_vectorstore_node` - Loads FAISS vectorstore
   - `instantiate_retriever_node` - Creates base retriever

2. **Query Generation**:
   - `generate_multi_queries_node` - Generates alternative queries using MultiQueryRetriever

3. **Document Retrieval**:
   - `extract_documents_node` - Retrieves relevant documents for each file

4. **Processing Path** (conditional):
   - `generate_answers_node` - Generates individual answers per file
   - `format_raw_documents_node` - Formats raw documents (bypass mode)

5. **Synthesis**:
   - `combine_answers_node` - Combines answers into final response

## Common Bottlenecks

### 1. LLM Operations
- **Symptoms**: High latency in `generate_multi_queries_node`, `generate_answers_node`, or `combine_answers_node`
- **Causes**: Large context windows, complex prompts, rate limiting
- **Solutions**: 
  - Use smaller LLM models for query generation
  - Reduce context window size
  - Implement caching for repeated queries

### 2. Document Retrieval
- **Symptoms**: High latency in `extract_documents_node`
- **Causes**: Large vectorstore, complex similarity search, many files
- **Solutions**:
  - Optimize FAISS index parameters
  - Reduce number of files processed
  - Use more efficient embedding models

### 3. Initialization
- **Symptoms**: High latency in initialization nodes
- **Causes**: Large vectorstore loading, model downloading
- **Solutions**:
  - Pre-load models and vectorstore
  - Use model caching
  - Optimize vectorstore size

## Profiling Output

### Basic Profiling
```
🚀 Starting KnowAI Profiling Session
⏰ 2024-01-15 10:30:00
================================================================================
🔍 Question: List the vegetation management strategies in table format with citations
📁 Files: ['Arizona_Public_Service_2024.pdf', 'BC_Hydro_2020.pdf']
⚡ Detailed Response: False
🗄️  Vectorstore: /path/to/vectorstore
================================================================================

✅ Agent initialization: 2.3456s
🔄 Executing workflow...

📊 Workflow Results:
   Total workflow time: 15.6789s
   Generation length: 1247 characters
   Files processed: 2
   Individual answers: 2
```

### Detailed Profiling
```
📈 Detailed Performance Analysis:
================================================================================
Node                                Duration (s)    % of Total   Status     Metadata
--------------------------------------------------------------------------------
combine_answers_node                 8.2345         52.5%        ✅         files: 2
extract_documents_node               4.1234         26.3%        ✅         docs: 45
generate_multi_queries_node          2.3456         15.0%        ✅         queries: 5
generate_answers_node                1.2345         7.9%         ✅         files: 2
instantiate_embeddings_node          0.1234         0.8%         ✅         
instantiate_llm_large_node           0.0987         0.6%         ✅         
instantiate_llm_small_node           0.0876         0.5%         ✅         
load_vectorstore_node                0.0765         0.5%         ✅         
instantiate_retriever_node           0.0543         0.3%         ✅         
format_raw_documents_node            0.0234         0.1%         ✅         
--------------------------------------------------------------------------------
TOTAL                                15.6789        100.0%       

🚨 Primary Bottleneck: combine_answers_node (8.2345s, 52.5%)
🔶 Secondary Bottleneck: extract_documents_node (4.1234s, 26.3%)

💡 Performance Insights:
   • LLM operations: 9.4690s (60.4% of total)
   • Retrieval operations: 4.1234s (26.3% of total)
   • Initialization operations: 0.4405s (2.8% of total)
```

## Optimization Strategies

### 1. Reduce LLM Latency
```python
# Use smaller models for query generation
agent = KnowAIAgent(
    vectorstore_path=vectorstore_path,
    # Configure smaller models
)
```

### 2. Optimize Retrieval
```python
# Reduce number of chunks retrieved
agent = KnowAIAgent(
    vectorstore_path=vectorstore_path,
    k_chunks_retriever=10,  # Default is 20
)
```

### 3. Use Bypass Mode
```python
# Skip individual answer generation for faster processing
result = await agent.process_turn(
    user_question=question,
    selected_files=files,
    bypass_individual_gen=True,  # Skip individual answers
    detailed_response_desired=False  # Use smaller LLM
)
```

### 4. Cache Results
```python
# Implement caching for repeated queries
# (Implementation depends on your caching strategy)
```

## Troubleshooting

### Common Issues

1. **Vectorstore not found**
   - Ensure the vectorstore path is correct
   - Check that the FAISS index files exist

2. **Azure OpenAI errors**
   - Verify environment variables are set
   - Check API quotas and rate limits

3. **Memory issues**
   - Reduce `k_chunks_retriever` value
   - Process fewer files at once

### Debug Mode

Enable debug logging for more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Custom Profiling

You can create custom profiling scripts by:

1. Using the `progress_cb` parameter in `process_turn()`
2. Implementing your own timing logic
3. Adding custom metrics

Example:
```python
def custom_progress_callback(node_name: str, status: str, metadata: dict):
    print(f"Node: {node_name}, Status: {status}, Metadata: {metadata}")

result = await agent.process_turn(
    user_question=question,
    selected_files=files,
    progress_cb=custom_progress_callback
)
```

## Performance Benchmarks

Typical performance for the test configuration:

- **Small vectorstore** (< 1GB): 5-10 seconds
- **Medium vectorstore** (1-5GB): 10-20 seconds  
- **Large vectorstore** (> 5GB): 20+ seconds

These times vary based on:
- Hardware specifications
- Network latency to Azure OpenAI
- Vectorstore size and complexity
- Number of files processed 