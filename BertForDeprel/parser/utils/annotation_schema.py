from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Self, Tuple

from conllup.conllup import _featuresJsonToConll, featuresJson_T, sentenceJson_T

from .lemma_script_utils import gen_lemma_script

NONE_VOCAB = "_none"  # default fallback

DUMMY_ID = -1


def _compute_labels2i(list_labels: Iterable[str]):
    sorted_set_labels = sorted(set(list_labels))

    labels2i: Dict[str, int] = {}

    for i, labels in enumerate(sorted_set_labels):
        labels2i[labels] = i

    return labels2i


def _update_mapping(
    existing_mapping: Dict[str, int],
    existing_list: List[str],
    new_list: List[str],
):
    new_labels = set(new_list) - set(existing_list)
    new_mapping = _compute_labels2i(new_labels)
    new_indices_start = len(existing_list)
    for label, index in new_mapping.items():
        # Add the new labels to the existing mapping; the indices must start
        # where the previous ones left off so that we don't have any collisions
        existing_mapping[label] = index + new_indices_start


@dataclass
class AnnotationSchema_T:
    deprels: List[str] = field(default_factory=list)
    uposs: List[str] = field(default_factory=list)
    xposs: List[str] = field(default_factory=list)
    feats: List[str] = field(default_factory=list)
    lemma_scripts: List[str] = field(default_factory=list)
    lemma_script_allow_copy: bool = False

    def __post_init__(self):
        self.dep2i = _compute_labels2i(self.deprels)
        self.upos2i = _compute_labels2i(self.uposs)
        self.xpos2i = _compute_labels2i(self.xposs)
        self.feat2i = _compute_labels2i(self.feats)
        self.lem2i = _compute_labels2i(self.lemma_scripts)

    @staticmethod
    def from_dict(schema_dict: Mapping[str, Any]):
        annotation_schema = AnnotationSchema_T()
        # TODO: Check the validity of this first; at least a version number
        annotation_schema.__dict__.update(schema_dict)
        annotation_schema.__post_init__()
        return annotation_schema

    def update(self, other: Self):
        _update_mapping(self.dep2i, self.deprels, other.deprels)
        _update_mapping(self.upos2i, self.uposs, other.uposs)
        _update_mapping(self.xpos2i, self.xposs, other.xposs)
        _update_mapping(self.feat2i, self.feats, other.feats)
        _update_mapping(self.lem2i, self.lemma_scripts, other.lemma_scripts)

    def is_empty(self):
        return not (self.uposs or self.deprels)

    def encode_deprel(self, deprel: str, word: str) -> int:
        return self._get_index(deprel, self.dep2i, word)

    def encode_upos(self, upos: str, word: str) -> int:
        return self._get_index(upos, self.upos2i, word)

    def encode_xpos(self, xpos: str, word: str) -> int:
        return self._get_index(xpos, self.xpos2i, word)

    def encode_feats(self, feats: featuresJson_T, word: str) -> int:
        return self._get_index(_featuresJsonToConll(feats), self.feat2i, word)

    def encode_lemma_script(self, form: str, lemma: str) -> int:
        return self._get_index(
            gen_lemma_script(form, lemma, self.lemma_script_allow_copy),
            self.lem2i,
            form,
        )

    def _get_index(self, label: str, mapping: Dict[str, int], word: str) -> int:
        """
        label: a string that represent the label whose integer is required
        mapping: a dictionary with a set of labels as keys and index integers as values
        word: the word having the label assigned to it (for logging purposes)
        return : index (int)
        """
        index = mapping.get(label, DUMMY_ID)

        if index == DUMMY_ID:
            index = mapping[NONE_VOCAB]
            print(
                f"LOG: label '{label}' for word '{word}' was not found in the "
                f"label2index mapping. Using the index for '{NONE_VOCAB}' instead."
            )
        return index


def choose_lemma_scripts(
    lemma_scripts_no_copy: List[str], lemma_scripts_allow_copy: List[str]
) -> Tuple[List[str], bool]:
    lemma_scripts_no_copy = sorted(set(lemma_scripts_no_copy))
    lemma_scripts_allow_copy = sorted(set(lemma_scripts_allow_copy))
    # if lengths are equal, use the one with shorter scripts
    if len(lemma_scripts_no_copy) < len(lemma_scripts_allow_copy) or (
        len(lemma_scripts_no_copy) == len(lemma_scripts_allow_copy)
        and len("".join(lemma_scripts_no_copy)) < len("".join(lemma_scripts_allow_copy))
    ):
        return lemma_scripts_no_copy, False
    else:
        return lemma_scripts_allow_copy, True


def compute_annotation_schema(
    sentences: Iterable[sentenceJson_T],
) -> AnnotationSchema_T:
    uposs: List[str] = []
    xposs: List[str] = []
    feats: List[str] = []
    deprels: List[str] = []
    lemma_scripts_no_copy: List[str] = []
    lemma_scripts_allow_copy: List[str] = []
    for sentence_json in sentences:
        for token in sentence_json["treeJson"]["nodesJson"].values():
            deprels.append(token["DEPREL"])
            uposs.append(token["UPOS"])
            xposs.append(token["XPOS"])
            feats.append(_featuresJsonToConll(token["FEATS"]))

            # TODO: per the literature, we should generate both with and without copy,
            # then keep the lemma scripts list that is shorter
            for scripts, allow_copy in [
                (lemma_scripts_no_copy, False),
                (lemma_scripts_allow_copy, True),
            ]:
                lemma_script = gen_lemma_script(
                    token["FORM"], token["LEMMA"], allow_copy=allow_copy
                )
                scripts.append(lemma_script)

    deprels.append(NONE_VOCAB)
    uposs.append(NONE_VOCAB)
    xposs.append(NONE_VOCAB)
    feats.append(NONE_VOCAB)
    lemma_scripts_no_copy.append(NONE_VOCAB)
    lemma_scripts_allow_copy.append(NONE_VOCAB)

    deprels = sorted(set(deprels))
    uposs = sorted(set(uposs))
    xposs = sorted(set(xposs))
    feats = sorted(set(feats))
    lemma_scripts, allow_copy = choose_lemma_scripts(
        lemma_scripts_no_copy, lemma_scripts_allow_copy
    )

    return AnnotationSchema_T(
        deprels=deprels,
        uposs=uposs,
        xposs=xposs,
        feats=feats,
        lemma_scripts=lemma_scripts,
        lemma_script_allow_copy=allow_copy,
    )
