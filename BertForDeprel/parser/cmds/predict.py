from argparse import ArgumentParser
import os
import sys
from typing import List
from conllup.conllup import writeConlluFile, sentenceJson_T
from timeit import default_timer as timer

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree # type: ignore (TODO: why can't PyLance find this?)
from torch import nn
from torch.utils.data import DataLoader


from ..cmds.cmd import CMD, SubparsersType
from ..utils.annotation_schema_utils import resolve_conllu_paths
from ..utils.load_data_utils import ConlluDataset, PartialPredictionConfig, SequencePredictionBatch_T
from ..modules.BertForDepRel import BertForDeprel
from ..utils.types import ModelParams_T


def max_span_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix_inverted = -1 * matrix
    max_span_inverted = minimum_spanning_tree(matrix_inverted)
    return -1 * (max_span_inverted.toarray().astype(int))


class Predict(CMD):
    def add_subparser(self, name: str, parser: SubparsersType) -> ArgumentParser:
        subparser = parser.add_parser(
            name, help="Use a trained model to make predictions."
        )
        subparser.add_argument("--inpath", '-i', required=True, help="path to inpath (can be a folder)")
        subparser.add_argument("--outpath", '-o',help="path to predicted outpath(s)")
        subparser.add_argument("--suffix", default="", help="suffix that will be added to the name of the predicted files (before the file extension)")
        subparser.add_argument(
            "--overwrite", action="store_true", help="whether to overwrite predicted file if already existing"
        )
        subparser.add_argument(
            "--write_preds_in_misc",
            action="store_true",
            help="whether to include punctuation",
        )
        subparser.add_argument(
            "--keep_heads", default="NONE",
            help="whether to use deps of input files as constrained for maximum spanning tree (NONE | EXISTING | ALL) (default : NONE)",
        )
        subparser.add_argument(
            "--keep_deprels", default="NONE", help="whether to keep current deprels and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )
        subparser.add_argument(
            "--keep_upos", default="NONE", help="whether to keep current upos and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )
        subparser.add_argument(
            "--keep_xpos", default="NONE", help="whether to keep current xpos and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )
        subparser.add_argument(
            "--keep_feats", default="NONE", help="whether to keep current feats and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )
        subparser.add_argument(
            "--keep_lemmas", default="NONE", help="whether to keep current lemmas and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )

        return subparser

    def __call__(self, args, model_params: ModelParams_T):
        super().__call__(args, model_params)
        in_to_out_paths, partial_pred_config, data_loader_params = self.__validate_args(args, model_params)
        model = self.__load_model(args, model_params)

        print("Starting Predictions ...")
        for in_path, out_path in in_to_out_paths.items():
            print(f"Loading dataset from {in_path}...")
            pred_dataset = ConlluDataset(in_path, model_params, "predict")

            pred_loader = DataLoader(pred_dataset, collate_fn=pred_dataset.collate_fn_predict, shuffle=False, **data_loader_params)
            print(
                f"Loaded {len(pred_dataset):5} sentences, ({len(pred_loader):3} batches)"
            )
            start = timer()
            predicted_sentences: List[sentenceJson_T] = []
            parsed_sentence_counter = 0
            batch: SequencePredictionBatch_T
            with torch.no_grad():
                for batch in pred_loader:
                    batch = batch.to(args.device)
                    # TODO: why is this detach()ed when we already have no_grad()?
                    preds = model.forward(batch).detach()

                    time_from_start = 0
                    parsing_speed = 0
                    for predicted_sentence in preds.iter():
                        predicted_sentences.append(predicted_sentence.get_predictions(partial_pred_config))

                        parsed_sentence_counter += 1
                        time_from_start = timer() - start
                        parsing_speed = int(round(((parsed_sentence_counter + 1) / time_from_start) / 100, 2) * 100)

                    print(f"Predicting: {100 * (parsed_sentence_counter + 1) / len(pred_dataset):.2f}% "
                          f"complete. {time_from_start:.2f} seconds in file "
                          f"({parsing_speed} sents/sec).")

            writeConlluFile(out_path, predicted_sentences, overwrite=args.overwrite)

            print(f"Finished predicting `{out_path}, wrote {parsed_sentence_counter} sents in {round(timer() - start, 2)} secs`")

    def __load_model(self, args, model_params):
        print("Loading model...")
        model = BertForDeprel(model_params)
        model.load_pretrained()
        model.eval()
        model.to(args.device)
        if args.multi_gpu:
            print("Sending model to multiple GPUs...")
            model = nn.DataParallel(model)
        return model

    def __validate_args(self, args, model_params: ModelParams_T):
        if not args.conf:
            raise Exception("Path to model xxx.config.json must be provided as --conf parameter")

        output_dir = args.outpath
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        unvalidated_input_paths = []
        if os.path.isdir(args.inpath):
            unvalidated_input_paths = resolve_conllu_paths(args.inpath)
        elif os.path.isfile(args.inpath):
            unvalidated_input_paths.append(args.inpath)
        else:
            raise BaseException(f"args.inpath must be a folder or a file; was {args.inpath}")

        in_to_out_paths = {}
        for input_path in unvalidated_input_paths:
            output_path = os.path.join(output_dir, input_path.split("/")[-1].replace(".conll", args.suffix + ".conll"))

            if args.overwrite != True:
                if os.path.isfile(output_path):
                    print(f"file '{output_path}' already exists and overwrite!=False, skipping ...", file=sys.stderr)
                    continue
            in_to_out_paths[input_path] = output_path

        partial_pred_config = PartialPredictionConfig(
            keep_upos=args.keep_upos,
            keep_xpos=args.keep_xpos,
            keep_feats=args.keep_feats,
            keep_deprels=args.keep_deprels,
            keep_heads=args.keep_heads,
            keep_lemmas=args.keep_lemmas
            )

        data_loader_params = {
            "batch_size": model_params.batch_size,
            "num_workers": args.num_workers,
        }

        if not in_to_out_paths:
            raise Exception("No legal input/output files determined.")

        return in_to_out_paths, partial_pred_config, data_loader_params
