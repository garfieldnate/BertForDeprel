import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from BertForDeprel.parser.cmds.predict import Predictor
from BertForDeprel.parser.cmds.train import Trainer
from BertForDeprel.parser.modules.BertForDepRel import BertForDeprel
from BertForDeprel.parser.utils.annotation_schema import compute_annotation_schema
from BertForDeprel.parser.utils.gpu_utils import get_devices_configuration
from BertForDeprel.parser.utils.load_data_utils import UDDataset, load_conllu_sentences
from BertForDeprel.parser.utils.types import ModelParams_T

PARENT = Path(__file__).parent
PATH_TEST_DATA_FOLDER = PARENT / "data"

PATH_TEST_CONLLU = PATH_TEST_DATA_FOLDER / "naija.test.conllu"
PATH_TRAIN_CONLLU = PATH_TEST_DATA_FOLDER / "naija.train.conllu"
PATH_EXPECTED_PREDICTIONS = PATH_TEST_DATA_FOLDER / "naija.predictions.expected.json"
PATH_MODELS_DIR = PATH_TEST_DATA_FOLDER / "models"


def _test_model_train():
    torch.manual_seed(42)
    model_config = ModelParams_T(
        model_folder_path=PATH_MODELS_DIR,
        max_epoch=1,
        patience=0,
        batch_size=16,
        max_position_embeddings=512,
        embedding_type="xlm-roberta-large",
    )

    device_config = get_devices_configuration("-1")

    train_sentences = load_conllu_sentences(PATH_TRAIN_CONLLU)
    model_config.annotation_schema = compute_annotation_schema(train_sentences)
    train_dataset = UDDataset(
        train_sentences,
        model_config.annotation_schema,
        model_config.embedding_type,
        model_config.max_position_embeddings,
        "train",
    )

    test_sentences = load_conllu_sentences(PATH_TEST_CONLLU)
    test_dataset = UDDataset(
        test_sentences,
        model_config.annotation_schema,
        model_config.embedding_type,
        model_config.max_position_embeddings,
        "train",
    )
    # TODO: we have to create train_dataset before calling Trainer()
    # because the former sets the annotation schema in our model config,
    # and the latter uses it to set up the model (reversing the ordering results in
    # errors related to length-0 tensors). That is really clumsy and annoying.
    # Next: fix this. Create model in separate step, and pass in training regime params
    # along with model to this constructor. Or something; maybe pass the training regime
    # params to a "train" method on the model.
    trainer = Trainer(model_config, device_config, overwrite_pretrain_classifiers=False)

    scores = list(trainer.train(train_dataset, test_dataset))

    # TODO: this API should take the output path as an argument
    trainer.model.save_model(1)  # type: ignore https://github.com/pytorch/pytorch/issues/81462 # noqa: E501

    # TODO: put time in result and check that, as well; or specify deadline to pytest
    # TODO: these numbers are different on every machine, and therefore this test FAILS
    # anywhere except for mine. Need to figure out how to make it pass anywhere.
    assert scores == pytest.approx(
        [
            {
                "LAS_epoch": 0.0,
                "LAS_chuliu_epoch": 0.0,
                "acc_head_epoch": 0.077,
                "acc_deprel_epoch": 0.0,
                "acc_uposs_epoch": 0.046,
                "acc_xposs_epoch": 1.0,
                "acc_feats_epoch": 0.0,
                "acc_lemma_scripts_epoch": 0.0,
                "loss_head_epoch": 3.045,
                "loss_deprel_epoch": 3.611,
                "loss_xposs_epoch": 0.531,
                "loss_feats_epoch": 3.367,
                "loss_lemma_scripts_epoch": 3.452,
                "loss_epoch": 17.016,
                "n_sentences_train": 39,
                "n_sentences_test": 5,
                "epoch": 0,
            },
            {
                "LAS_epoch": 0.015,
                "LAS_chuliu_epoch": 0.015,
                "acc_head_epoch": 0.123,
                "acc_deprel_epoch": 0.308,
                "acc_uposs_epoch": 0.046,
                "acc_xposs_epoch": 1.0,
                "acc_feats_epoch": 0.0,
                "acc_lemma_scripts_epoch": 0.0,
                "loss_head_epoch": 3.041,
                "loss_deprel_epoch": 3.37,
                "loss_xposs_epoch": 0.415,
                "loss_feats_epoch": 3.228,
                "loss_lemma_scripts_epoch": 3.133,
                "loss_epoch": 16.117,
                "n_sentences_train": 39,
                "n_sentences_test": 5,
                "epoch": 1,
            },
        ]
    )


def _get_model_config():
    with open(PATH_MODELS_DIR / "config.json", "r") as f:
        params_dict = json.load(f)
        return ModelParams_T.from_dict(params_dict)


def _test_predict():
    model_config = _get_model_config()

    predictor = Predictor(model_config, 1)

    sentences = load_conllu_sentences(PATH_TEST_CONLLU)
    pred_dataset = UDDataset(
        sentences,
        model_config.annotation_schema,
        model_config.embedding_type,
        model_config.max_position_embeddings,
        "train",
    )

    actual, elapsed_seconds = predictor.predict(pred_dataset)

    with open(PATH_EXPECTED_PREDICTIONS, "r") as f:
        expected = json.load(f)

    assert actual == expected
    # On my M2, it's <7s.
    if elapsed_seconds > 10:
        print(
            f"WARNING: Prediction took a long time: {elapsed_seconds} seconds.",
            file=sys.stderr,
        )


def _test_eval():
    """There is no eval API, per se, but this demonstrates how to do it. TODO: it's
    pretty convoluted."""
    model_config = _get_model_config()
    device_config = get_devices_configuration("-1")

    model = BertForDeprel(model_config)
    model.load_pretrained()
    model.eval()
    model = model.to(device_config.device)

    sentences = load_conllu_sentences(PATH_TEST_CONLLU)
    test_dataset = UDDataset(
        sentences,
        model_config.annotation_schema,
        model_config.embedding_type,
        model_config.max_position_embeddings,
        "train",
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn_train,
        batch_size=16,
        num_workers=1,
    )

    results = model.eval_on_dataset(test_loader, device_config.device)

    # TODO: these are different on each machine, and therefore this test FAILS anywhere
    # but mine.
    assert results == pytest.approx(
        {
            "LAS_epoch": 0.015,
            "LAS_chuliu_epoch": 0.015,
            "acc_head_epoch": 0.123,
            "acc_deprel_epoch": 0.308,
            "acc_uposs_epoch": 0.046,
            "acc_xposs_epoch": 1.0,
            "acc_feats_epoch": 0.0,
            "acc_lemma_scripts_epoch": 0.0,
            "loss_head_epoch": 3.041,
            "loss_deprel_epoch": 3.37,
            "loss_xposs_epoch": 0.415,
            "loss_feats_epoch": 3.228,
            "loss_lemma_scripts_epoch": 3.133,
            "loss_epoch": 16.117,
        }
    )


# About 30s on my M2 Mac.
@pytest.mark.slow
@pytest.mark.fragile
def test_train_and_predict():
    _test_model_train()
    _test_predict()
    _test_eval()
