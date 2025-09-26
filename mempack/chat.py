"""Optional MemPackChat helper for LLM integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .retriever import MemPackRetriever
from .types import SearchHit
from .logging import cli_logger


class MemPackChat:
    """Chat interface using MemPack for context retrieval."""
    
    def __init__(
        self,
        retriever: MemPackRetriever,
        context_chunks: int = 8,
        max_context_length: int = 2000,
    ) -> None:
        """Initialize the chat interface.
        
        Args:
            retriever: MemPack retriever
            context_chunks: Number of chunks to use as context
            max_context_length: Maximum context length in characters
        """
        self.retriever = retriever
        self.context_chunks = context_chunks
        self.max_context_length = max_context_length
        
        # Session state
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_id: Optional[str] = None
    
    def start_session(self, session_id: Optional[str] = None) -> None:
        """Start a new chat session.
        
        Args:
            session_id: Optional session identifier
        """
        self.session_id = session_id or f"session_{len(self.conversation_history)}"
        self.conversation_history = []
        cli_logger.info(f"Started chat session: {self.session_id}")
    
    def chat(
        self,
        user_input: str,
        llm_client: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Process user input and return response.
        
        Args:
            user_input: User's message
            llm_client: Optional LLM client for response generation
            system_prompt: Optional system prompt
            
        Returns:
            Chat response
        """
        if not user_input.strip():
            return "Please provide a message to chat about."
        
        # Search for relevant context
        hits = self.retriever.search(
            query=user_input,
            top_k=self.context_chunks,
        )
        
        # Build context from search results
        context = self._build_context(hits)
        
        # Generate response
        if llm_client is not None:
            response = self._generate_llm_response(
                user_input=user_input,
                context=context,
                llm_client=llm_client,
                system_prompt=system_prompt,
            )
        else:
            response = self._generate_simple_response(
                user_input=user_input,
                context=context,
            )
        
        # Store in conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "context_sources": [hit.meta.get("source", "unknown") for hit in hits],
            "context_scores": [hit.score for hit in hits],
        })
        
        return response
    
    def _build_context(self, hits: List[SearchHit]) -> str:
        """Build context string from search hits.
        
        Args:
            hits: Search results
            
        Returns:
            Context string
        """
        if not hits:
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        for i, hit in enumerate(hits):
            # Add source information
            source = hit.meta.get("source", f"Document {i+1}")
            context_part = f"Source: {source}\n{hit.text}\n\n"
            
            # Check if adding this part would exceed max length
            if current_length + len(context_part) > self.max_context_length:
                break
            
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "".join(context_parts)
    
    def _generate_llm_response(
        self,
        user_input: str,
        context: str,
        llm_client: Any,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response using LLM client.
        
        Args:
            user_input: User's message
            context: Context from knowledge pack
            llm_client: LLM client
            system_prompt: System prompt
            
        Returns:
            Generated response
        """
        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Use the context information to provide accurate and helpful responses. 
            If the context doesn't contain relevant information, say so politely."""
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ]
        
        try:
            # Call LLM (this is a generic interface - adapt to your LLM client)
            if hasattr(llm_client, 'chat_completion'):
                response = llm_client.chat_completion(messages)
            elif hasattr(llm_client, 'generate'):
                response = llm_client.generate(messages)
            else:
                # Fallback to simple response
                return self._generate_simple_response(user_input, context)
            
            return response
            
        except Exception as e:
            cli_logger.warning(f"LLM generation failed: {e}")
            return self._generate_simple_response(user_input, context)
    
    def _generate_simple_response(
        self,
        user_input: str,
        context: str,
    ) -> str:
        """Generate a simple response without LLM.
        
        Args:
            user_input: User's message
            context: Context from knowledge pack
            
        Returns:
            Simple response
        """
        if not context or context == "No relevant context found.":
            return "I don't have enough information to answer that question. Could you try asking something else?"
        
        # Simple response based on context
        context_preview = context[:500] + "..." if len(context) > 500 else context
        
        return f"""Based on the information I found:

{context_preview}

Is there anything specific you'd like to know more about?"""
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history.copy()
    
    def export_session(self, path: Union[str, Path]) -> None:
        """Export conversation session to file.
        
        Args:
            path: Path to export file
        """
        path = Path(path)
        
        session_data = {
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "context_chunks": self.context_chunks,
            "max_context_length": self.max_context_length,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        cli_logger.info(f"Session exported to {path}")
    
    def load_session(self, path: Union[str, Path]) -> None:
        """Load conversation session from file.
        
        Args:
            path: Path to session file
        """
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        self.session_id = session_data.get("session_id")
        self.conversation_history = session_data.get("conversation_history", [])
        self.context_chunks = session_data.get("context_chunks", self.context_chunks)
        self.max_context_length = session_data.get("max_context_length", self.max_context_length)
        
        cli_logger.info(f"Session loaded from {path}")
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        cli_logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "session_id": self.session_id,
            "conversation_turns": len(self.conversation_history),
            "context_chunks": self.context_chunks,
            "max_context_length": self.max_context_length,
        }
