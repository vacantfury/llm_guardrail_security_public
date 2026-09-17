"""
Set theory processor - MathPrompt attack using set theory, abstract algebra, and symbolic logic.

Prompts, few-shot demonstrations, and target prefix are loaded from:
  conf/text_encoding/set_theory.yaml

Reference: MathPrompt paper (Figures 3-5, Appendix A.4)
"""
from typing import Optional, List
from llm_utils import LLMServiceFactory, BaseLLMService
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder, strip_delimiter_tags
from ..prompt_loader import load_prompt_template


logger = get_logger(__name__)


class SetTheoryLLMEncoder(BaseEncoder):
    """
    MathPrompt attack strategy using set theory, abstract algebra, and symbolic logic.
    
    All prompts are loaded from conf/text_encoding/set_theory.yaml.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_few_shot: bool = True,
        **kwargs
    ):
        if model is None:
            raise ValueError("model is required — set it in conf/text_encoding/default.yaml")
        
        super().__init__(model=model, **kwargs)
        self.use_few_shot = use_few_shot
        
        # Load prompts from YAML
        config = load_prompt_template("set_theory.yaml")
        self.system_prompt = config["system_prompt"].strip()
        self.user_template = config["user_template"].strip()
        self.target_prefix = config.get("target_prefix", "").strip()
        self._few_shot_data = config.get("few_shot", [])
        
        # Create LLM service
        self.service: BaseLLMService = LLMServiceFactory.create(model)
        if temperature is not None:
            self.service.temperature = temperature
        if max_tokens is not None:
            self.service.max_tokens = max_tokens
        
        logger.info(f"Initialized SetTheoryLLMEncoder with model: {model}, use_few_shot: {use_few_shot}")
    
    # Keep TARGET_PREFIX as a property for backward compatibility
    @property
    def TARGET_PREFIX(self):
        return self.target_prefix + "\n\n" if self.target_prefix else ""
    
    @TARGET_PREFIX.setter
    def TARGET_PREFIX(self, value):
        # Allow base class __init__ to set via target_prefix override
        self.target_prefix = value.strip() if value else ""
    
    def _get_few_shot_demonstrations(self) -> list:
        """Get few-shot demonstrations from YAML config."""
        return [
            (demo["user"], demo["assistant"].strip())
            for demo in self._few_shot_data
        ]
    
    def process(self, prompt: str, **kwargs) -> str:
        """Transform harmful prompt into mathematical problem using MathPrompt approach."""
        user_message = self.user_template.format(prompt=prompt)
        
        if self.use_few_shot:
            few_shot_demos = self._get_few_shot_demonstrations()
            few_shot_context = "\n\n".join([
                f"User:\n{user}\n\nAssistant:\n{assistant}"
                for user, assistant in few_shot_demos
            ])
            full_prompt = f"{few_shot_context}\n\nUser:\n{user_message}\n\nAssistant:"
            
            conversations = [("single", [(full_prompt, None)])]
            results = self.service.batch_chat(
                conversations=conversations,
                system_message=self.system_prompt
            )
        else:
            conversations = [("single", [(user_message, None)])]
            results = self.service.batch_chat(
                conversations=conversations,
                system_message=self.system_prompt
            )
        
        if results and len(results) > 0:
            _, response = results[0]
            return strip_delimiter_tags(response)
        
        return "Error: No response generated"
    
    def _batch_process_core(self, prompts: List[str], **kwargs) -> List[str]:
        """Process multiple prompts efficiently by sending batch request to LLM API."""
        logger.info(f"Batch processing {len(prompts)} prompts via LLM API")
        
        batch_prompts = []
        few_shot_context = None
        
        if self.use_few_shot:
            few_shot_demos = self._get_few_shot_demonstrations()
            few_shot_context = "\n\n".join([
                f"User:\n{user}\n\nAssistant:\n{assistant}"
                for user, assistant in few_shot_demos
            ])
        
        for i, prompt in enumerate(prompts):
            user_message = self.user_template.format(prompt=prompt)
            
            if self.use_few_shot:
                full_prompt = f"{few_shot_context}\n\nUser:\n{user_message}\n\nAssistant:"
            else:
                full_prompt = user_message
            
            batch_prompts.append((str(i), full_prompt))
        
        logger.info(f"Sending batch of {len(batch_prompts)} transformation requests to API")
        conversations = [(pid, [(text, None)]) for pid, text in batch_prompts]
        batch_results = self.service.batch_chat(
            conversations=conversations,
            system_message=self.system_prompt
        )
        
        result_dict = {prompt_id: response for prompt_id, response in batch_results}
        processed_prompts = []
        
        for i in range(len(prompts)):
            response = result_dict.get(str(i), "Error: No response generated")
            processed_prompts.append(response)
        
        logger.info(f"Batch processing complete: {len(processed_prompts)} mathematical formulations generated")
        return [strip_delimiter_tags(p) for p in processed_prompts]