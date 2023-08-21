import json
import sys
from pathlib import Path
from typing import List

import pytest
import torch
from conllup.conllup import emptyNodeJson, emptySentenceJson, sentenceJson_T
from torch.utils.data import DataLoader

from BertForDeprel.parser.cmds.predict import Predictor
from BertForDeprel.parser.cmds.train import Trainer
from BertForDeprel.parser.modules.BertForDepRel import BertForDeprel, EvalResult
from BertForDeprel.parser.utils.annotation_schema import compute_annotation_schema
from BertForDeprel.parser.utils.gpu_utils import get_devices_configuration
from BertForDeprel.parser.utils.load_data_utils import load_conllu_sentences
from BertForDeprel.parser.utils.types import (
    ModelParams_T,
    PredictionConfig,
    TrainingConfig,
)

PARENT = Path(__file__).parent
PATH_TEST_DATA_FOLDER = PARENT / "data"
PATH_MODELS_DIR = PATH_TEST_DATA_FOLDER / "models"

PATH_TRAIN_NAIJA = PATH_TEST_DATA_FOLDER / "naija.train.conllu"
PATH_TEST_NAIJA = PATH_TEST_DATA_FOLDER / "naija.test.conllu"
PATH_EXPECTED_PREDICTIONS_NAIJA = (
    PATH_TEST_DATA_FOLDER / "naija.predictions.expected.json"
)
NAIJA_MODEL_DIR = PATH_MODELS_DIR / "naija"

PATH_TEST_ENGLISH = PATH_TEST_DATA_FOLDER / "english.test.conllu"
PATH_TRAIN_ENGLISH = PATH_TEST_DATA_FOLDER / "english.train.conllu"
PATH_EXPECTED_PREDICTIONS_ENGLISH = (
    PATH_TEST_DATA_FOLDER / "english.predictions.expected.json"
)
ENGLISH_MODEL_DIR = PATH_MODELS_DIR / "english"
SEED = 42


def _test_model_train_single(path_train, path_test, path_out, expected_eval):
    train_sentences = load_conllu_sentences(path_train)
    annotation_schema = compute_annotation_schema(train_sentences)

    device_config = get_devices_configuration("-1")

    # for reproducibility we need to set the seed during training; for example,
    # nn.Dropout uses rng during training time to drop a layer's weights, but at test
    # time it doesn't drop anything, so there's no random behavior.
    torch.manual_seed(SEED)
    model = BertForDeprel.new_model(
        "xlm-roberta-large", annotation_schema, device_config.device
    )
    print(model.max_position_embeddings)

    train_dataset = model.encode_dataset(train_sentences)
    test_sentences = load_conllu_sentences(path_test)
    test_dataset = model.encode_dataset(test_sentences)
    training_config = TrainingConfig(
        max_epochs=1,
        patience=0,
    )
    trainer = Trainer(
        training_config,
        device_config.multi_gpu,
    )

    scores_generator = trainer.train(model, train_dataset, test_dataset)
    scores = [next(scores_generator), next(scores_generator)]
    scores = [s.rounded(3) for s in scores]

    model.save_model(  # type: ignore https://github.com/pytorch/pytorch/issues/81462 # noqa: E501
        path_out, training_config
    )

    assert scores == pytest.approx(expected_eval)


def _test_model_train():
    _test_model_train_single(
        PATH_TRAIN_NAIJA,
        PATH_TEST_NAIJA,
        NAIJA_MODEL_DIR,
        [
            EvalResult(
                LAS_epoch=0.0,
                LAS_chuliu_epoch=0.0,
                acc_head_epoch=0.077,
                acc_deprel_epoch=0.0,
                acc_uposs_epoch=0.046,
                acc_xposs_epoch=1.0,
                acc_feats_epoch=0.0,
                acc_lemma_scripts_epoch=0.0,
                loss_head_epoch=0.609,
                loss_deprel_epoch=0.722,
                loss_uposs_epoch=0.602,
                loss_xposs_epoch=0.106,
                loss_feats_epoch=0.673,
                loss_lemma_scripts_epoch=0.69,
                loss_epoch=0.567,
                training_diagnostics=None,
            ),
            EvalResult(
                LAS_epoch=0.015,
                LAS_chuliu_epoch=0.015,
                acc_head_epoch=0.123,
                acc_deprel_epoch=0.308,
                acc_uposs_epoch=0.046,
                acc_xposs_epoch=1.0,
                acc_feats_epoch=0.0,
                acc_lemma_scripts_epoch=0.0,
                loss_head_epoch=0.608,
                loss_deprel_epoch=0.674,
                loss_uposs_epoch=0.586,
                loss_xposs_epoch=0.083,
                loss_feats_epoch=0.646,
                loss_lemma_scripts_epoch=0.627,
                loss_epoch=0.537,
                training_diagnostics=None,
            ),
        ],
    )
    _test_model_train_single(
        PATH_TRAIN_ENGLISH,
        PATH_TEST_ENGLISH,
        ENGLISH_MODEL_DIR,
        [
            EvalResult(
                LAS_epoch=0.0,
                LAS_chuliu_epoch=0.0,
                acc_head_epoch=0.08,
                acc_deprel_epoch=0.008,
                acc_uposs_epoch=0.088,
                acc_xposs_epoch=1.0,
                acc_feats_epoch=0.008,
                acc_lemma_scripts_epoch=0.008,
                loss_head_epoch=0.322,
                loss_deprel_epoch=0.318,
                loss_uposs_epoch=0.258,
                loss_xposs_epoch=0.037,
                loss_feats_epoch=0.341,
                loss_lemma_scripts_epoch=0.298,
                loss_epoch=0.262,
            ),
            EvalResult(
                LAS_epoch=0.008,
                LAS_chuliu_epoch=0.008,
                acc_head_epoch=0.088,
                acc_deprel_epoch=0.28,
                acc_uposs_epoch=0.104,
                acc_xposs_epoch=1.0,
                acc_feats_epoch=0.008,
                acc_lemma_scripts_epoch=0.144,
                loss_head_epoch=0.321,
                loss_deprel_epoch=0.295,
                loss_uposs_epoch=0.251,
                loss_xposs_epoch=0.026,
                loss_feats_epoch=0.332,
                loss_lemma_scripts_epoch=0.268,
                loss_epoch=0.249,
            ),
        ],
    )


def _test_predict_single(
    predictor: Predictor, input: List[sentenceJson_T], expected: Path, max_seconds: int
):
    actual, elapsed_seconds = predictor.predict(input)

    with open(expected, "r") as f:
        expected = json.load(f)

    with open("actual.json", "w") as f:
        json.dump(actual, f, indent=2)

    assert actual == expected
    # On my M2, it's <7s.
    if elapsed_seconds > max_seconds:
        print(
            f"WARNING: Prediction took a long time: {elapsed_seconds} seconds.",
            file=sys.stderr,
        )


def _test_predict():
    model_config = ModelParams_T.from_model_path(NAIJA_MODEL_DIR)
    device_config = get_devices_configuration("-1")

    model = BertForDeprel.load_pretrained_for_prediction(
        {"naija": NAIJA_MODEL_DIR, "english": ENGLISH_MODEL_DIR},
        "naija",
        device_config.device,
    )
    predictor = Predictor(
        model,
        PredictionConfig(batch_size=model_config.batch_size, num_workers=1),
        device_config.multi_gpu,
    )

    naija_sentences = load_conllu_sentences(PATH_TEST_NAIJA)
    # add a sentence too large for the model; this should be skipped in the output
    too_long = emptySentenceJson()
    too_long["metaJson"]["sent_id"] = "too_long"
    for i in range(model.max_position_embeddings + 10):
        too_long["treeJson"]["nodesJson"][f"{i}"] = emptyNodeJson(ID=f"{i}")
    naija_sentences.insert(2, too_long)

    # On my M1, it's <7s.
    _test_predict_single(
        predictor, naija_sentences, PATH_EXPECTED_PREDICTIONS_NAIJA, 10
    )

    # model.activate("english")
    # english_sentences = load_conllu_sentences(PATH_TEST_ENGLISH)
    # _test_predict_single(
    #     predictor, english_sentences, PATH_EXPECTED_PREDICTIONS_ENGLISH, 10
    # )


def _test_eval():
    """There is no eval API, per se, but this demonstrates how to do it. TODO: it's
    pretty convoluted."""
    device_config = get_devices_configuration("-1")

    model = BertForDeprel.load_single_pretrained_for_prediction(
        NAIJA_MODEL_DIR, device_config.device
    )

    sentences = load_conllu_sentences(PATH_TEST_NAIJA)
    test_dataset = model.encode_dataset(sentences)
    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_train,
        batch_size=16,
        num_workers=1,
    )

    results = model.eval_on_dataset(test_loader)

    # TODO: these are different on each machine, and therefore this test FAILS anywhere
    # but mine.
    assert results.rounded(3) == pytest.approx(
        EvalResult(
            LAS_epoch=0.015,
            LAS_chuliu_epoch=0.015,
            acc_head_epoch=0.123,
            acc_deprel_epoch=0.308,
            acc_uposs_epoch=0.046,
            acc_xposs_epoch=1.0,
            acc_feats_epoch=0.0,
            acc_lemma_scripts_epoch=0.0,
            loss_head_epoch=0.608,
            loss_deprel_epoch=0.674,
            loss_uposs_epoch=0.586,
            loss_xposs_epoch=0.083,
            loss_feats_epoch=0.646,
            loss_lemma_scripts_epoch=0.627,
            loss_epoch=0.537,
            training_diagnostics=None,
        ),
    )


# About 30s on my M2 Mac.
@pytest.mark.slow
@pytest.mark.fragile
def test_train_and_predict():
    # _test_model_train()
    _test_predict()
    # _test_eval()
