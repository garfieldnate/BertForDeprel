import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional


@dataclass
class AnnotationSchema_T:
    deprels: List[str] = field(default_factory=list)
    uposs: List[str] = field(default_factory=list)
    xposs: List[str] = field(default_factory=list)
    feats: List[str] = field(default_factory=list)
    lemma_scripts: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(schema_dict: Mapping[str, Any]):
        annotation_schema = AnnotationSchema_T()
        # TODO: Check the validity of this first; at least a version number
        annotation_schema.__dict__.update(schema_dict)
        return annotation_schema


@dataclass
class ModelParams_T:
    # Shared
    # TODO: what behavior does an empty path lead to?
    model_folder_path: Optional[Path] = None
    # TODO: what behavior does an empty schema lead to?
    annotation_schema: AnnotationSchema_T = field(default_factory=AnnotationSchema_T)

    # Training params
    # In our experiments, most of the models based on UD data converged in 10-15 epochs
    max_epoch: int = 15
    # How many epochs with no performance improvement before training is ended early
    patience: int = 3
    # How many sentences to process in each batch
    batch_size: int = 16

    # Finetuned training meta params
    # n_current_epoch: int
    # current_epoch_results: EpochResults_T

    # Allows a copy command in the lemma scripts. In the UDPipe paper, they tried both
    # with and without this option and kept the one that yielded fewer unique scripts.
    allow_lemma_char_copy: bool = False

    # Pre-trained embeddings to download from 🤗 (xlm-roberta-large /
    # bert-multilingual-base-uncased ...)
    embedding_type: str = "xlm-roberta-large"
    # Maximum length of an input sequence; the default value is the default from
    # xlm-roberta-large. Using larger values could result in doubling or quadrupling the
    # memory usage.
    max_position_embeddings: int = 512

    @staticmethod
    def from_dict(params_dict: Mapping[str, Any]) -> "ModelParams_T":
        model_params = ModelParams_T()
        model_params.__dict__.update(params_dict)
        if model_params.model_folder_path:
            model_params.model_folder_path = Path(model_params.model_folder_path)
        annotation_schema = AnnotationSchema_T()
        # TODO: Check the validity of this first; at least a version number
        annotation_schema.__dict__.update(params_dict["annotation_schema"])
        model_params.annotation_schema = annotation_schema

        return model_params


class ConfigJSONEncoder(json.JSONEncoder):
    """JSON encoder for data that may include dataclasses."""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, Path):
            return str(o)
        return super().default(o)
