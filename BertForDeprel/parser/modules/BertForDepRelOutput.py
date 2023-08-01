from dataclasses import dataclass
from typing import List, Tuple
import torch

from ..utils.chuliu_edmonds_utils import chuliu_edmonds_one_root_with_constraints
from ..utils.lemma_script_utils import apply_lemma_rule
from ..utils.load_data_utils import DUMMY_ID, ConlluDataset, CopyOption, PartialPredictionConfig
from ..utils.scores_and_losses_utils import _deprel_pred_for_heads

from conllup.conllup import _featuresConllToJson, sentenceJson_T

@dataclass
class BertForDeprelSentenceOutput:
    """Prediction tensors for uposs, xposs, feats, and lemma_scripts have size (T, C),
    where T is the maximum sequence length for the containing batch, and C is the
    number of classes being assigned a probability for the particular tensor.

    deprels tensor has size (C, T, T), with the first T corresponding to
    the dependent token, and the second T corresponding to the head token.

    heads tensor has size (T, T), with the first T corresponding to the
    dependent token, and the second T containing the head score for each
    potential head of the dependent.
    """
    uposs: torch.Tensor
    xposs: torch.Tensor
    feats: torch.Tensor
    lemma_scripts: torch.Tensor
    deprels: torch.Tensor
    heads: torch.Tensor

    # 1 if sequence token begins a new word, 0 otherwise. Size is (B, T).
    subwords_start: torch.Tensor
    # Maps word index + 1 to the index in the sequence_token_ids where the word begins. Size is (W).
    idx_converter: torch.Tensor

    # Index of the sentence in the original dataset.
    idx: int
    dataset: ConlluDataset


    def get_predictions(self, partial_pred_config: PartialPredictionConfig) -> sentenceJson_T:
        # TODO Next: encapsulate below in the output classes;
        # these will then be containers for the raw model outputs, with methods for constructing the final predictions.
        # the overwrite logic should be done in a separate step, I think.
        # Start by encapsulating the n_sentence and pred_dataset stuff into the output classes.
        # sentence_idx = batch.idx[i_sentence]
        # n_sentence = int(sentence_idx)

        (chuliu_heads_list, deprels_pred_chuliu) = self.__get_constrained_dependencies(
            keep_heads=partial_pred_config.keep_heads,
        )

        # Indices for words in the sentence (not subwords)
        mask = self.subwords_start == 1

        deprels_pred_chuliu_list = deprels_pred_chuliu.max(dim=0).indices[
            mask
        ].tolist()

        uposs_pred_list = self.uposs.max(dim=1).indices[
            mask
        ].tolist()

        xposs_pred_list = self.xposs.max(dim=1).indices[
            mask
        ].tolist()

        feats_pred_list = self.feats.max(dim=1).indices[
            mask
        ].tolist()

        lemma_scripts_pred_list = self.lemma_scripts.max(dim=1).indices[
            mask
        ].tolist()

        return self.construct_sentence_prediction(
            int(self.idx),
            uposs_pred_list,
            xposs_pred_list,
            chuliu_heads_list,
            deprels_pred_chuliu_list,
            feats_pred_list,
            lemma_scripts_pred_list,
            partial_pred_config=partial_pred_config
        )

    # TODO Next: explain current return type. Return type differs from method in training eval.
    # Can we refactor to join them?
    def __get_constrained_dependencies(self, keep_heads: CopyOption):
        """TODO"""
        # TODO: explain these
        head_true_like = self.heads.max(dim=0).indices
        chuliu_heads_pred = head_true_like.clone().cpu().numpy()
        chuliu_heads_list: List[int] = []
        subwords_start = self.subwords_start.clone()

        # Keep the rows and columns corresponding to tokens that begin words
        # (which we use to represent entire words). Size is (W + 1, W + 1)
        # (+1 is for dummy root).
        heads_pred_np = self.heads[
            :, subwords_start
        ][subwords_start]
        # Chu-Liu/Edmonds code needs a Numpy array, which can only be created from the CPU device
        heads_pred_np = heads_pred_np.cpu().numpy()

        forced_relations: List[Tuple] = []
        if keep_heads == "EXISTING":
            forced_relations = self.dataset.get_specified_heads_for_chuliu(int(self.idx))

        # Size is (W + 1,)
        chuliu_heads_vector = chuliu_edmonds_one_root_with_constraints(
            heads_pred_np, forced_relations
        )
        # Ignore head predicted for dummy root (CLS token); size is (W,)
        chuliu_heads_vector = chuliu_heads_vector[1:]

        # Convert from word indices to token indices
        for i_dependent_word, chuliu_head_pred in enumerate(chuliu_heads_vector):
            chuliu_heads_pred[
                self.idx_converter[i_dependent_word + 1]
            ] = self.idx_converter[chuliu_head_pred]
            chuliu_heads_list.append(int(chuliu_head_pred))

        # move to device where deprels_pred lives so they can be operated on together
        # in _deprel_pred_for_heads
        chuliu_heads_pred = torch.tensor(chuliu_heads_pred).to(self.deprels.device)

        # function is designed to work with batch inputs;
        # unsqueeze to add dummy batch dimension to fit expected input shape, and
        # squeeze to remove dummy batch dimension from output shape
        deprels_pred_chuliu = _deprel_pred_for_heads(
            self.deprels.unsqueeze(0), chuliu_heads_pred.unsqueeze(0)
        ).squeeze(0)

        # TODO: what are these return values?
        return chuliu_heads_list, deprels_pred_chuliu


    def construct_sentence_prediction(self,
                                        idx,
                                        uposs_preds: List[int]=[],
                                        xposs_preds: List[int]=[],
                                        chuliu_heads: List[int]=[],
                                        deprels_pred_chulius: List[int]=[],
                                        feats_preds: List[int]=[],
                                        lemma_scripts_preds: List[int]=[],
                                        partial_pred_config = PartialPredictionConfig(),
                                        ) -> sentenceJson_T:
        """Constructs the final sentence structure by 1) converting the provided class indices to
        textual values, 2) overwriting the predictions with the input data where specified, and
        3) copying the metadata from the original input."""
        predicted_sentence: sentenceJson_T = self.dataset.sequences[idx].sentence_json.copy()
        tokens = list(predicted_sentence["treeJson"]["nodesJson"].values())
        annotation_schema = self.dataset.model_params.annotation_schema

        # For each of the predicted fields, we overwrite the value copied from the input with the predicted value
        # if configured to do so.
        for n_token, token in enumerate(tokens):
            if partial_pred_config.keep_upos=="NONE" or (partial_pred_config.keep_upos=="EXISTING" and token["UPOS"] == "_"):
                token["UPOS"] = annotation_schema.uposs[uposs_preds[n_token]]

            if partial_pred_config.keep_xpos == "NONE" or (partial_pred_config.keep_xpos=="EXISTING" and token["XPOS"] == "_"):
                token["XPOS"] = annotation_schema.xposs[xposs_preds[n_token]]

             # this one is special as for keep_heads == "EXISTING", we already handled the case earlier in the code
            if partial_pred_config.keep_heads == "NONE" or (partial_pred_config.keep_heads == "EXISTING" and token["HEAD"] == DUMMY_ID):
                token["HEAD"] = chuliu_heads[n_token]

            if partial_pred_config.keep_deprels == "NONE" or (partial_pred_config.keep_deprels=='EXISTING' and token["DEPREL"] == "_"):
                token["DEPREL"] = annotation_schema.deprels[deprels_pred_chulius[n_token]]

            if partial_pred_config.keep_feats == "NONE" or (partial_pred_config.keep_feats=="EXISTING" and token["FEATS"] == {}):
                token["FEATS"] = _featuresConllToJson(annotation_schema.feats[feats_preds[n_token]])

            if partial_pred_config.keep_lemmas == "NONE" or (partial_pred_config.keep_lemmas=="EXISTING" and token["LEMMA"] == "_"):
                lemma_script = annotation_schema.lemma_scripts[lemma_scripts_preds[n_token]]
                token["LEMMA"] = apply_lemma_rule(token["FORM"], lemma_script)

        return predicted_sentence

@dataclass
class BertForDeprelBatchOutput:
    """For each field of BertForDeprelSentenceOutput, each corresponding field here has
    the same size, but with an additional leading dimension for the batch size B.
    For example, the size of uposs is (B, T, C), and the size of heads is (B, T, T)."""
    uposs: torch.Tensor
    xposs: torch.Tensor
    feats: torch.Tensor
    lemma_scripts: torch.Tensor
    deprels: torch.Tensor
    heads: torch.Tensor

    # 1 if sequence token begins a new word, 0 otherwise. Size is (B, T).
    subwords_start: torch.Tensor
    # Maps word index + 1 to the index in the sequence_token_ids where the word begins. Size is (B, W).
    idx_converter: torch.Tensor

    # Size (B,). Index of each sentence in the original dataset.
    idx: torch.Tensor

    dataset: ConlluDataset

    def detach(self):
        """Return a new result with all of the Tensors detached from backprop (used for prediction)."""
        return BertForDeprelBatchOutput(
            uposs=self.uposs.detach(),
            xposs=self.xposs.detach(),
            feats=self.feats.detach(),
            lemma_scripts=self.lemma_scripts.detach(),
            deprels=self.deprels.detach(),
            heads=self.heads.detach(),
            subwords_start=self.subwords_start,
            idx_converter=self.idx_converter,
            idx=self.idx,
            dataset=self.dataset
        )

    def results_for_sentence(self, idx_in_batch: int) -> BertForDeprelSentenceOutput:
        """Return the model output for the sentence at the specified index."""
        # TODO: why is cloning necessary here?
        # TODO: maybe encapsulate the raw stuff by itself, then introduce a clone method for that
        return BertForDeprelSentenceOutput(
            heads=self.heads[idx_in_batch].clone(),
            deprels=self.deprels[idx_in_batch].clone(),
            uposs=self.uposs[idx_in_batch].clone(),
            xposs=self.xposs[idx_in_batch].clone(),
            feats= self.feats[idx_in_batch].clone(),
            lemma_scripts=self.lemma_scripts[idx_in_batch].clone(),
            subwords_start = self.subwords_start[idx_in_batch],
            idx_converter = self.idx_converter[idx_in_batch],
            idx=int(self.idx[idx_in_batch]),
            dataset=self.dataset
        )

    def iter(self):
        for idx_in_batch in range(self.idx.size()[0]):
            yield self.results_for_sentence(idx_in_batch)
