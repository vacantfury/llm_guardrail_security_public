"""
Base processor class and helper functions for all prompt encoders.
"""
import re
from abc import ABC, abstractmethod
from typing import Optional, List

from llm_utils import LLMModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Delimiter tags used by LLM processors to mark encoding vs instruction sections
ENCODING_START_TAG = "[ENCODING]"
ENCODING_END_TAG = "[/ENCODING]"
INSTRUCTION_START_TAG = "[INSTRUCTION]"
INSTRUCTION_END_TAG = "[/INSTRUCTION]"

# Repeat transition phrase
REPEAT_TRANSITION = "\n"


class BaseEncoder(ABC):
    """
    Abstract base class for all prompt encoders.
    
    Processors transform prompts using different techniques (LLM-based or rule-based).
    Each processor must implement process() for single prompt transformation.
    The default batch_process() uses multiprocessing for efficiency.
    
    Built-in pre/post processing:
    - rephrase_first: If True, rephrases prompts via RephraseLLMEncoder before core processing.
    - is_repeating: If True, repeats the output after core processing.
    
    All processors accept an optional 'model' parameter for consistency,
    even if they don't use it (e.g., rule-based processors).
    
    If no model is provided for LLM-based processors, they will use 
    DEFAULT_PROCESSING_MODEL from encoders/constants.py (GPT-4o).
    """
    
    # Default prefix prepended to processed prompts when sent to target model.
    # Can be overridden via YAML config (target_prefix kwarg) or subclass attribute.
    TARGET_PREFIX: str = ""
    
    def __init__(
        self,
        model: Optional[LLMModel] = None,
        rephrase_first: bool = False,
        is_repeating: bool = False,
        rephrase_model: Optional[LLMModel] = None,
        target_prefix: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the processor.
        
        Args:
            model: Optional LLM model (used by LLM-based processors).
                  If None, LLM-based processors will use DEFAULT_PROCESSING_MODEL.
            rephrase_first: If True, pre-process prompts through RephraseLLMEncoder
                           before the core processing logic runs.
            is_repeating: If True, repeat the entire output after core processing.
            rephrase_model: Optional model for rephrasing. If None, uses DEFAULT_PROCESSING_MODEL.
            target_prefix: Override for TARGET_PREFIX (loaded from YAML config).
                          If provided, takes precedence over the class attribute.
            **kwargs: Encoder-specific parameters
        """
        self.model = model
        self.rephrase_first = rephrase_first
        self.is_repeating = is_repeating
        self.rephrase_model = rephrase_model
        self._rephraser = None  # Lazy initialization
        
        # YAML target_prefix overrides class-level TARGET_PREFIX
        if target_prefix is not None:
            self.TARGET_PREFIX = target_prefix.strip() + "\n\n"

    def get_usage(self) -> Optional[dict]:
        """Return LLM usage stats if this encoder uses an LLM service, else None."""
        service = getattr(self, "service", None)
        if service and hasattr(service, "get_usage"):
            return service.get_usage()
        return None

    @abstractmethod
    def process(self, prompt: str, **kwargs) -> str:
        """
        Process a single prompt (core logic).
        
        Args:
            prompt: The prompt to process
            **kwargs: Encoder-specific parameters
            
        Returns:
            Processed prompt or response
        """
        pass
    
    def _apply_repeat(self, output: str) -> str:
        """
        Apply repeat post-processing to a single output.
        
        Args:
            output: The processed output to repeat
            
        Returns:
            Output with repetition applied
        """
        if self.is_repeating:
            return output + REPEAT_TRANSITION + output
        return output
    
    def _get_rephraser(self):
        """Lazily create the rephrase processor."""
        if self._rephraser is None:
            from .encoders.llm_rephrase_encoder import RephraseLLMEncoder
            self._rephraser = RephraseLLMEncoder(model=self.rephrase_model)
            logger.info(f"Initialized rephraser for pre-processing (model: {self.rephrase_model})")
        return self._rephraser
    
    def batch_process(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Process multiple prompts with optional pre-step (rephrase) and post-step (repeat).
        
        Pipeline: [rephrase] -> core processing -> [repeat]
        
        Args:
            prompts: List of prompts to process
            **kwargs: Encoder-specific parameters
            
        Returns:
            List of processed prompts/responses
        """
        if not prompts:
            logger.warning("No prompts to process")
            return []
        
        # Pre-step: rephrase
        if self.rephrase_first:
            logger.info(f"Pre-step: Rephrasing {len(prompts)} prompts before core processing")
            rephraser = self._get_rephraser()
            prompts = rephraser._batch_process_core(prompts, **kwargs)
            logger.info("Pre-step complete: Rephrasing done")
        
        # Core processing
        results = self._batch_process_core(prompts, **kwargs)
        
        # Post-step: repeat
        if self.is_repeating:
            logger.info(f"Post-step: Applying repeat to {len(results)} outputs")
            results = [self._apply_repeat(r) for r in results]
            logger.info("Post-step complete: Repeat applied")
        
        # Prepend TARGET_PREFIX so the returned prompts are the complete
        # text that should be sent to the target model.
        if self.TARGET_PREFIX:
            logger.info(f"Prepending TARGET_PREFIX ({len(self.TARGET_PREFIX)} chars) to processed prompts")
            results = [self.TARGET_PREFIX + r for r in results]
        
        return results
    
    def _batch_process_core(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Core batch processing logic (without pre/post steps).
        
        LLM-based processors should override this to use API batch calls.
        
        Args:
            prompts: List of prompts to process
            **kwargs: Encoder-specific parameters
            
        Returns:
            List of processed prompts/responses
        """
        logger.info(f"Batch processing {len(prompts)} prompts (sequential)")
        results = self._sequential_process(prompts, **kwargs)
        logger.info(f"Batch processing complete: {len(results)} results")
        return results
    
    def _sequential_process(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Process prompts sequentially (fallback or small batches).
        
        Args:
            prompts: List of prompts to process
            **kwargs: Encoder-specific parameters
            
        Returns:
            List of processed prompts
        """
        results = []
        for prompt in prompts:
            try:
                result = self.process(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing prompt: {str(e)}")
                results.append(f"Error: {str(e)}")
        return results
    
    def _process_with_error_handling(self, prompt: str, **kwargs) -> str:
        """
        Wrapper for parallel processing with error handling.
        
        Args:
            prompt: The prompt to process
            **kwargs: Encoder-specific parameters
            
        Returns:
            Processed result or error message
        """
        try:
            return self.process(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            return f"Error: {str(e)}"


def strip_delimiter_tags(text: str) -> str:
    """
    Remove [ENCODING], [/ENCODING], [INSTRUCTION], [/INSTRUCTION] tags from text.
    
    Used by LLM processors to clean model output before returning.
    """
    text = text.replace(ENCODING_START_TAG, "")
    text = text.replace(ENCODING_END_TAG, "")
    text = text.replace(INSTRUCTION_START_TAG, "")
    text = text.replace(INSTRUCTION_END_TAG, "")
    # Clean up any resulting double-newlines from tag removal
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def split_into_parts(words: list[str], num_parts: int) -> list[list[str]]:
    """
    Split a list of words into parts with smart boundary detection.
    
    This function tries to split at natural boundaries (sentences, phrases)
    rather than cutting words arbitrarily.
    
    Args:
        words: List of words to split
        num_parts: Number of parts to create
        
    Returns:
        List of parts, where each part is a list of words
    """
    # Import constants locally to avoid circular imports
    from .encoders.constants import (
        BOUNDARY_WINDOW_PERCENT,
        SENTENCE_BOUNDARY_SCORE,
        PHRASE_BOUNDARY_SCORE,
        WORD_BOUNDARY_SCORE,
        MIN_CONTENT_WORD_LENGTH,
        SENTENCE_ENDERS,
        PHRASE_ENDERS
    )
    
    if len(words) <= num_parts:
        return [[word] for word in words[:num_parts]]
    
    # Calculate target size for each part
    target_size = len(words) / num_parts
    
    parts = []
    current_idx = 0
    
    for i in range(num_parts - 1):
        target_end = int((i + 1) * target_size)
        end_idx = target_end
        
        # Look for a good boundary within a reasonable range
        window_size = max(1, int(target_size * BOUNDARY_WINDOW_PERCENT))
        best_boundary = end_idx
        best_score = -1
        
        # Search within the window for an optimal boundary
        search_start = max(current_idx + 1, end_idx - window_size)
        search_end = min(len(words) - 1, end_idx + window_size)
        
        for j in range(search_start, search_end + 1):
            if j <= current_idx or j >= len(words):
                continue
            
            current_word = words[j - 1]
            score = 0
            
            # Prefer sentence boundaries over phrase boundaries
            if any(current_word.endswith(end) for end in SENTENCE_ENDERS):
                score = SENTENCE_BOUNDARY_SCORE
            elif any(current_word.endswith(end) for end in PHRASE_ENDERS):
                score = PHRASE_BOUNDARY_SCORE
            elif len(current_word) > MIN_CONTENT_WORD_LENGTH:
                score = WORD_BOUNDARY_SCORE
            
            if score > best_score:
                best_score = score
                best_boundary = j
        
        # Use the best boundary found
        if best_score > 0:
            end_idx = best_boundary
        
        parts.append(words[current_idx:end_idx])
        current_idx = end_idx
    
    # Add the last part (remaining words)
    parts.append(words[current_idx:])
    
    return parts
