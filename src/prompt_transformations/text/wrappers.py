"""
Thin PromptTransformation wrappers around legacy BaseEncoder implementations.

Each class is a 3-line subclass that:
  - declares its canonical type_name (registry key)
  - points at its legacy encoder class

The factory registers each one via @register_transformation.
"""
from src.prompt_transformations.transformation_factory import register_transformation
from .base_transformation import TextEncoderTransformation
from .encoders.non_llm_baseline_encoder import BaselineEncoder
from .encoders.non_llm_addition_equation_split_reassemble_encoder import (
    AdditionEquationEncoder,
)
from .encoders.non_llm_conditional_probability_encoder import (
    ConditionalProbabilityEncoder,
)
from .encoders.non_llm_symbol_injection_encoder import SymbolInjectionEncoder
from .encoders.llm_set_theory_encoder import SetTheoryLLMEncoder
from .encoders.llm_formal_logic_encoder import FormalLogicLLMEncoder
from .encoders.llm_quantum_mechanics_encoder import QuantumMechanicsLLMEncoder
from .encoders.llm_classical_language_encoder import LLMClassicalLanguageEncoder
from .encoders.llm_semantic_camo_encoder import SemanticCamoEncoder
from .encoders.llm_decode_evasion_encoder import DecodeEvasionEncoder
from .encoders.non_llm_artprompt_encoder import ArtPromptEncoder
from .encoders.non_llm_homoglyph_encoder import HomoglyphEncoder
from .encoders.non_llm_cipher_encoder import CipherEncoder
from .encoders.non_llm_best_of_n_encoder import BestOfNEncoder
from .encoders.llm_paraphrase_encoder import ParaphraseLLMEncoder


@register_transformation
class BaselineTransformation(TextEncoderTransformation):
    type_name = "non_llm_baseline"
    encoder_class = BaselineEncoder


@register_transformation
class ArtPromptTransformation(TextEncoderTransformation):
    type_name = "non_llm_artprompt"
    encoder_class = ArtPromptEncoder


@register_transformation
class HomoglyphTransformation(TextEncoderTransformation):
    type_name = "non_llm_homoglyph"
    encoder_class = HomoglyphEncoder


@register_transformation
class CipherTransformation(TextEncoderTransformation):
    type_name = "non_llm_cipher"
    encoder_class = CipherEncoder


@register_transformation
class BestOfNTransformation(TextEncoderTransformation):
    type_name = "non_llm_best_of_n"
    encoder_class = BestOfNEncoder


@register_transformation
class ParaphraseTransformation(TextEncoderTransformation):
    # Stochastic meaning-preserving paraphrase — the "which-paraphrase" variance
    # knob for the Variance-Channel Best-of-N attack (Paper D).
    type_name = "llm_paraphrase"
    encoder_class = ParaphraseLLMEncoder


@register_transformation
class AdditionEquationTransformation(TextEncoderTransformation):
    type_name = "non_llm_addition_equation_split_reassemble"
    encoder_class = AdditionEquationEncoder


@register_transformation
class ConditionalProbabilityTransformation(TextEncoderTransformation):
    type_name = "non_llm_conditional_probability"
    encoder_class = ConditionalProbabilityEncoder


@register_transformation
class SymbolInjectionTransformation(TextEncoderTransformation):
    type_name = "non_llm_symbol_injection"
    encoder_class = SymbolInjectionEncoder


@register_transformation
class SetTheoryTransformation(TextEncoderTransformation):
    type_name = "llm_set_theory"
    encoder_class = SetTheoryLLMEncoder


@register_transformation
class FormalLogicTransformation(TextEncoderTransformation):
    type_name = "llm_formal_logic"
    encoder_class = FormalLogicLLMEncoder


@register_transformation
class QuantumMechanicsTransformation(TextEncoderTransformation):
    type_name = "llm_quantum_mechanics"
    encoder_class = QuantumMechanicsLLMEncoder


@register_transformation
class ClassicalLanguageTransformation(TextEncoderTransformation):
    type_name = "llm_classical_language"
    encoder_class = LLMClassicalLanguageEncoder


@register_transformation
class SemanticCamoTransformation(TextEncoderTransformation):
    type_name = "llm_semantic_camo"
    encoder_class = SemanticCamoEncoder


@register_transformation
class DecodeEvasionTransformation(TextEncoderTransformation):
    # Adaptive attack (Paper C RQ4): reframe so the amplifier's 1-2 sentence decode
    # reads benign while the full answer stays harmful. See llm_decode_evasion_encoder.py.
    type_name = "llm_decode_evasion"
    encoder_class = DecodeEvasionEncoder
