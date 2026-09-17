"""
Classical Language encoder — translates prompts into classical languages.

Supports multiple languages (Classical Chinese, Latin, Sanskrit, Old English)
and two strategies:
  - direct:   Simple translation (CC-BOS ablation "CC" only, ~18% ASR for Chinese)
  - literary: Translation with multidimensional literary strategy
              (CC-BOS ablation "CC + Strategy", ~60% ASR for Chinese)

Special case: classical_chinese_traditional uses the same prompts as
classical_chinese (simplified), with deterministic opencc conversion
to traditional characters as post-processing.

Reference:
  Huang et al. (2026). "Obscure but Effective: Classical Chinese Jailbreak
  Prompt Optimization via Bio-Inspired Search." ICLR 2026.
"""
from typing import Optional, List
from llm_utils import LLMServiceFactory, BaseLLMService
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder
from ..prompt_loader import load_prompt_template


logger = get_logger(__name__)

# Supported languages (must have a corresponding YAML in classical_language/)
SUPPORTED_LANGUAGES = {
    "classical_chinese_simplified", "classical_chinese_traditional",
    "latin", "sanskrit", "old_english",
}
SUPPORTED_STRATEGIES = {"direct", "literary"}

# Languages that use another language's prompts + post-processing
_LANGUAGE_ALIASES = {
    "classical_chinese_traditional": "classical_chinese_simplified",
}


def _convert_simplified_to_traditional(text: str) -> str:
    """Convert simplified Chinese to traditional Chinese via opencc."""
    try:
        import opencc
        converter = opencc.OpenCC("s2t")
        return converter.convert(text)
    except ImportError:
        logger.warning(
            "opencc-python-reimplemented not installed. "
            "Install with: pip install opencc-python-reimplemented. "
            "Returning simplified text as fallback.")
        return text


class LLMClassicalLanguageEncoder(BaseEncoder):
    """
    Encodes prompts into classical languages using an LLM writer model.
    
    Parameterized by:
      - language: which classical language to translate into
      - strategy: "direct" (simple translation) or "literary" (with literary dimensions)
    
    Prompt templates are loaded from:
      conf/text_encoding/classical_language/{language}.yaml
    """
    
    def __init__(
        self,
        language: str = "classical_chinese",
        strategy: str = "direct",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize classical language encoder.
        
        Args:
            language: Target language. One of: classical_chinese,
                classical_chinese_traditional, latin, sanskrit, old_english.
            strategy: Encoding strategy.
                - "direct": Simple translation (1 LLM call, no literary framing)
                - "literary": Translation with multidimensional literary strategy
                  (CC-BOS's 8 dimensions: role, mechanism, metaphor, etc.)
            model: LLM model for translation (default: GPT-4o).
            temperature: Sampling temperature (None = service default).
            max_tokens: Max tokens to generate (None = service default).
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: '{language}'. "
                f"Supported: {SUPPORTED_LANGUAGES}")
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy: '{strategy}'. "
                f"Supported: {SUPPORTED_STRATEGIES}")
        
        if model is None:
            raise ValueError("model is required — set it in conf/text_encoding/default.yaml")
        
        super().__init__(model=model, **kwargs)
        self.language = language
        self.strategy = strategy
        
        # Resolve alias (e.g., classical_chinese_traditional → classical_chinese)
        self._base_language = _LANGUAGE_ALIASES.get(language, language)
        self._needs_s2t = (language == "classical_chinese_traditional")
        
        # Load prompt template from YAML (using base language for aliases)
        template_data = load_prompt_template(
            f"classical_language/{self._base_language}.yaml")
        
        if strategy not in template_data:
            raise ValueError(
                f"Strategy '{strategy}' not found in template for '{self._base_language}'. "
                f"Available: {list(template_data.keys())}")
        
        self.template = template_data[strategy]
        self.system_prompt = self.template["system_prompt"].strip()
        self.user_template = self.template["user_template"].strip()
        
        # For literary strategy, load dimension values
        self.dimensions = self.template.get("dimensions", {})
        
        # Create LLM service
        self.service: BaseLLMService = LLMServiceFactory.create(model)
        if temperature is not None:
            self.service.temperature = temperature
        if max_tokens is not None:
            self.service.max_tokens = max_tokens
        
        s2t_note = " (→ traditional via opencc)" if self._needs_s2t else ""
        logger.info(
            f"Initialized LLMClassicalLanguageEncoder: "
            f"language={language}, strategy={strategy}, model={model}{s2t_note}")
    
    def _postprocess(self, text: str) -> str:
        """Apply any post-processing (e.g., simplified → traditional conversion)."""
        if self._needs_s2t:
            return _convert_simplified_to_traditional(text)
        return text
    
    def process(self, prompt: str, **kwargs) -> str:
        """
        Encode a single prompt into the target classical language.
        
        Args:
            prompt: The English prompt to encode.
            
        Returns:
            The prompt translated/encoded into the classical language.
        """
        # Build user message from template
        if self.strategy == "literary" and self.dimensions:
            # Fill in dimension slots
            user_msg = self.user_template.format(
                prompt=prompt, **self.dimensions)
        else:
            user_msg = self.user_template.format(prompt=prompt)
        
        # Single LLM call
        response = self.service.generate(
            prompt=user_msg,
            system_message=self.system_prompt,
        )
        
        return self._postprocess(response.strip())
    
    def _batch_process_core(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Batch process using the LLM service's batch API.
        """
        logger.info(
            f"Batch encoding {len(prompts)} prompts → {self.language} "
            f"(strategy={self.strategy})")
        
        # Build batch of (id, prompt) tuples
        batch = []
        for i, prompt in enumerate(prompts):
            if self.strategy == "literary" and self.dimensions:
                user_msg = self.user_template.format(
                    prompt=prompt, **self.dimensions)
            else:
                user_msg = self.user_template.format(prompt=prompt)
            batch.append((str(i), user_msg))
        
        # Call LLM batch API
        conversations = [(pid, [(text, None)]) for pid, text in batch]
        results = self.service.batch_chat(
            conversations=conversations,
            system_message=self.system_prompt,
        )
        
        # Sort by index and extract responses, apply post-processing
        result_dict = {pid: self._postprocess(response.strip())
                       for pid, response in results}
        return [result_dict.get(str(i), f"Error: no response for {i}")
                for i in range(len(prompts))]

