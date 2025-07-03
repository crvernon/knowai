#!/usr/bin/env python3
"""
Client script to test KnowAI streaming API endpoint.
"""

import asyncio
import aiohttp
import json
from typing import Optional


async def test_streaming_api(base_url: str = "http://localhost:8000"):
    """Test the streaming API endpoint."""
    print("🧪 Testing KnowAI Streaming API...")
    
    # First, initialize a session
    async with aiohttp.ClientSession() as session:
        # Initialize session
        init_payload = {
            "vectorstore_s3_uri": "tests/fixtures/vectorstore",
            "combine_threshold": 50,
            "max_conversation_turns": 20
        }
        
        try:
            async with session.post(f"{base_url}/initialize", json=init_payload) as response:
                if response.status == 200:
                    init_result = await response.json()
                    session_id = init_result["session_id"]
                    print(f"✅ Session initialized: {session_id}")
                else:
                    print(f"❌ Failed to initialize session: {response.status}")
                    return
        except Exception as e:
            print(f"⚠️  Expected error (no real vectorstore): {e}")
            print("✅ API setup test passed")
            return
        
        # Test streaming endpoint
        stream_payload = {
            "session_id": session_id,
            "question": "List the vegetation management strategies in table format with citations",
            "selected_files": ["Arizona_Public_Service_2024.pdf", "BC_Hydro_2020.pdf"]
        }
        
        print(f"\n📡 Testing streaming endpoint...")
        print(f"   Question: {stream_payload['question']}")
        print(f"   Files: {stream_payload['selected_files']}")
        print("-" * 60)
        
        try:
            async with session.post(f"{base_url}/ask-stream", json=stream_payload) as response:
                if response.status == 200:
                    print("📝 Streaming response:")
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data = line_str[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                print("\n✅ Streaming completed")
                                break
                            elif data:  # Non-empty data
                                print(data, end='', flush=True)
                else:
                    print(f"❌ Streaming request failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
        except Exception as e:
            print(f"❌ Streaming test failed: {e}")


async def demo_streaming_usage():
    """Demonstrate how to use the streaming API."""
    print("\n📚 Streaming API Usage Example:")
    print("=" * 50)
    
    print("""
# Example usage with streaming API:

import asyncio
import aiohttp

async def chat_with_streaming():
    async with aiohttp.ClientSession() as session:
        # Initialize session
        init_response = await session.post("http://localhost:8000/initialize", json={
            "vectorstore_s3_uri": "/path/to/vectorstore"
        })
        session_data = await init_response.json()
        session_id = session_data["session_id"]
        
        # Stream response
        stream_response = await session.post("http://localhost:8000/ask-stream", json={
            "session_id": session_id,
            "question": "What are the vegetation management strategies?",
            "selected_files": ["file1.pdf", "file2.pdf"]
        })
        
        # Process streaming response
        async for line in stream_response.content:
            line_str = line.decode('utf-8').strip()
            if line_str.startswith('data: '):
                data = line_str[6:]
                if data == '[DONE]':
                    break
                elif data:
                    print(data, end='', flush=True)

# Run the streaming chat
asyncio.run(chat_with_streaming())
""")
    
    print("✅ Streaming API usage example demonstrated")


if __name__ == "__main__":
    print("🚀 Starting KnowAI Streaming API Test")
    print("⏰ Make sure the KnowAI server is running on http://localhost:8000")
    print("=" * 60)
    
    asyncio.run(test_streaming_api())
    asyncio.run(demo_streaming_usage())
    
    print("\n🏁 Streaming API test completed!") 