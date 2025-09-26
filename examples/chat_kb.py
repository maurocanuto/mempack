#!/usr/bin/env python3
"""Example: Simple chat interface with knowledge pack."""

import sys
from pathlib import Path
from typing import List

# Add mempack to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mempack import MemPackRetriever


class SimpleChat:
    """Simple chat interface using MemPack."""
    
    def __init__(self, retriever: MemPackRetriever):
        """Initialize the chat interface.
        
        Args:
            retriever: MemPack retriever
        """
        self.retriever = retriever
        self.conversation_history = []
    
    def chat(self, user_input: str) -> str:
        """Process user input and return response.
        
        Args:
            user_input: User's message
            
        Returns:
            Chat response
        """
        # Search for relevant chunks
        hits = self.retriever.search(user_input, top_k=3)
        
        if not hits:
            return "I don't have information about that topic. Could you try asking something else?"
        
        # Build context from search results
        context = []
        for hit in hits:
            context.append(f"Source: {hit.meta.get('source', 'unknown')}\n{hit.text}")
        
        # Simple response generation (in a real implementation, you'd use an LLM)
        response = self._generate_response(user_input, context)
        
        # Store in conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "sources": [hit.meta.get('source', 'unknown') for hit in hits]
        })
        
        return response
    
    def _generate_response(self, query: str, context: List[str]) -> str:
        """Generate a simple response based on context.
        
        Args:
            query: User query
            context: Relevant context chunks
            
        Returns:
            Generated response
        """
        # This is a very simple response generator
        # In a real implementation, you'd use an LLM like OpenAI's GPT
        
        if "quantum" in query.lower():
            return f"""Based on the information I found:

{context[0][:300]}...

Quantum computing is a fascinating field that could revolutionize computation. Would you like to know more about specific aspects of quantum computing?"""
        
        elif "machine learning" in query.lower() or "ml" in query.lower():
            return f"""Here's what I found about machine learning:

{context[0][:300]}...

Machine learning is a powerful subset of AI. Are you interested in learning about specific algorithms or applications?"""
        
        elif "python" in query.lower():
            return f"""Here are some Python programming tips:

{context[0][:300]}...

Python is a great language for many applications. Would you like to know about specific Python features or best practices?"""
        
        else:
            return f"""I found some relevant information:

{context[0][:300]}...

Is there anything specific you'd like to know more about?"""
    
    def get_conversation_history(self) -> List[dict]:
        """Get conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history


def main():
    """Run the chat interface."""
    print("MemPack Chat Interface")
    print("=" * 40)
    print("Type 'quit' to exit, 'history' to see conversation history")
    print()
    
    # Check if knowledge pack exists
    pack_path = Path("example_kb.mpack")
    ann_path = Path("example_kb.ann")
    
    if not pack_path.exists() or not ann_path.exists():
        print("Error: Knowledge pack not found!")
        print("Please run build_kb.py first to create the example knowledge pack.")
        return
    
    # Create retriever and chat interface
    with MemPackRetriever(
        pack_path=pack_path,
        ann_path=ann_path,
        ef_search=32,
    ) as retriever:
        
        chat = SimpleChat(retriever)
        
        print("Chat started! Ask me anything about the knowledge pack.")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'history':
                    history = chat.get_conversation_history()
                    if history:
                        print("\nConversation History:")
                        print("-" * 30)
                        for i, turn in enumerate(history, 1):
                            print(f"{i}. You: {turn['user']}")
                            print(f"   Me: {turn['assistant'][:100]}...")
                            print(f"   Sources: {', '.join(turn['sources'])}")
                            print()
                    else:
                        print("No conversation history yet.")
                    continue
                elif not user_input:
                    continue
                
                # Get response
                response = chat.chat(user_input)
                print(f"Assistant: {response}")
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
