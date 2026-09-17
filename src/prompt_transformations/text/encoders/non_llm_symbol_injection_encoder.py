"""
Symbol injection processor.

Injects mathematical and special symbols randomly and evenly into prompts
to obfuscate content while maintaining readability.
"""
import random
from typing import Optional, List

from llm_utils import LLMModel
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder
from .constants import DEFAULT_INJECTION_SYMBOLS

logger = get_logger(__name__)


class SymbolInjectionEncoder(BaseEncoder):
    """
    Inject mathematical and special symbols randomly and evenly into prompts.
    
    This processor takes a prompt and injects symbols from a predefined list
    at evenly distributed positions throughout the text. The symbols are chosen
    randomly from the provided symbol list.
    
    Example:
        Input:  "How to break into a secure system"
        Output: "How ∀ to break ∈ into ∑ a secure ∇ system"
        
    Parameters:
        num_symbols: Number of symbols to inject (default: 10)
        symbols: List of symbols to choose from (default: mathematical symbols)
    """

    
    def __init__(
        self,
        model: Optional[LLMModel] = None,
        num_symbols: int = 10,
        symbols: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the symbol injection processor.
        
        Args:
            model: Unused (for consistency with other processors)
            num_symbols: Number of symbols to inject into each prompt
            symbols: List of symbols to randomly choose from.
                    If None, uses DEFAULT_INJECTION_SYMBOLS.
            **kwargs: Additional parameters (unused)
        """
        super().__init__(model, **kwargs)
        self.num_symbols = num_symbols
        self.symbols = symbols if symbols is not None else DEFAULT_INJECTION_SYMBOLS
        
        logger.info(
            f"Initialized SymbolInjectionEncoder with {self.num_symbols} symbols "
            f"from a pool of {len(self.symbols)} available symbols"
        )
    
    def process(self, prompt: str, **kwargs) -> str:
        """
        Inject symbols randomly and evenly into the prompt.
        
        Args:
            prompt: The original prompt text
            **kwargs: Additional parameters (unused)
            
        Returns:
            The prompt with symbols injected at evenly distributed positions
        """
        if not prompt:
            logger.warning("Empty prompt provided")
            return prompt
        
        if not self.symbols:
            logger.warning("No symbols provided, returning original prompt")
            return prompt
        
        if self.num_symbols <= 0:
            logger.warning("num_symbols <= 0, returning original prompt")
            return prompt
        
        # Split prompt into words for injection
        words = prompt.split()
        
        if len(words) <= 1:
            # If prompt is too short, just append a symbol
            if self.num_symbols > 0:
                selected_symbol = random.choice(self.symbols)
                return f"{prompt} {selected_symbol}"
            return prompt
        
        # Calculate positions for even distribution
        # We'll inject symbols between words, so we have (len(words) - 1) possible positions
        num_positions = len(words) - 1
        
        # Limit num_symbols to available positions
        actual_num_symbols = min(self.num_symbols, num_positions)
        
        if actual_num_symbols == 0:
            logger.warning("Not enough positions to inject symbols")
            return prompt
        
        # Calculate evenly distributed positions
        # We want to spread symbols evenly across the prompt
        if actual_num_symbols == num_positions:
            # Inject after every word except the last
            injection_positions = list(range(num_positions))
        else:
            # Calculate step size for even distribution
            step = num_positions / actual_num_symbols
            injection_positions = [
                int(i * step) for i in range(actual_num_symbols)
            ]
        
        # Randomly select symbols for each position
        selected_symbols = [
            random.choice(self.symbols) for _ in range(actual_num_symbols)
        ]
        
        # Build the result by injecting symbols at calculated positions
        result_parts = []
        symbol_index = 0
        
        for i, word in enumerate(words):
            result_parts.append(word)
            
            # Check if we should inject a symbol after this word
            if symbol_index < len(injection_positions) and i == injection_positions[symbol_index]:
                result_parts.append(selected_symbols[symbol_index])
                symbol_index += 1
        
        result = " ".join(result_parts)
        
        logger.debug(
            f"Injected {actual_num_symbols} symbols into prompt "
            f"({len(words)} words → {len(result_parts)} parts)"
        )
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the processor."""
        return (
            f"SymbolInjectionEncoder("
            f"num_symbols={self.num_symbols}, "
            f"symbols_pool_size={len(self.symbols)})"
        )

