from dataclasses import dataclass
import torch


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
            idx=self.idx
        )

    def distributions_for_sentence(self, sentence_idx: int) -> BertForDeprelSentenceOutput:
        """Return the model output for the sentence at the specified index."""
        return BertForDeprelSentenceOutput(
            heads=self.heads[sentence_idx].clone(),
            deprels=self.deprels[sentence_idx].clone(),
            uposs=self.uposs[sentence_idx].clone(),
            xposs=self.xposs[sentence_idx].clone(),
            feats= self.feats[sentence_idx].clone(),
            lemma_scripts=self.lemma_scripts[sentence_idx].clone(),
            subwords_start = self.subwords_start[sentence_idx],
            idx_converter = self.idx_converter[sentence_idx],
            idx=int(self.idx[sentence_idx])
        )
