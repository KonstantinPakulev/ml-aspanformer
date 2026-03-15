import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_aspanformer import PL_ASpanFormer
import torch

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # Lightning 2.x: add_argparse_args is deprecated, manually add common Trainer flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default="weights/indoor_ds.ckpt", help='path to the checkpoint')
    parser.add_argument(
        '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument(
        '--dump_inputs_dir', type=str, default=None,
        help="if set, model inputs (image0/1, mask0/1) are saved as .pt files before the forward pass")
    parser.add_argument(
        '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--thr', type=float, default=None, help='modify the coarse-level matching threshold.')
    parser.add_argument(
        '--mode', type=str, default='vanilla', help='modify the coarse-level matching threshold.')
    parser.add_argument(
        '--max_pairs', type=int, default=None, help='stop after this many pairs (for debugging)')
    # Common Trainer flags for Lightning 2.x
    parser.add_argument('--gpus', type=str, default='-1', help='gpus to use, -1 for all')
    parser.add_argument('--benchmark', action='store_true', help='enable cuDNN benchmarking')
    # Lightning 2.x: add_argparse_args is deprecated, manually add common flags
    # parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    # Lightning 2.x: seed_everything is deprecated, use seed_everything from pl.utilities
    try:
        pl.seed_everything(config.TRAINER.SEED, workers=True)  # reproducibility
    except TypeError:
        # Fallback for older Lightning versions
        pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # tune when testing
    if args.thr is not None:
        config.ASPAN.MATCH_COARSE.THR = args.thr

    loguru_logger.info(f"Args and config initialized!")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_ASpanFormer(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir,
                           max_pairs=args.max_pairs, dump_inputs_dir=args.dump_inputs_dir)
    loguru_logger.info(f"ASpanFormer-lightning initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # lightning trainer
    # Lightning 2.x: from_argparse_args is deprecated, manually create Trainer
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    gpus = args.gpus if args.gpus != '-1' else -1
    if gpus == -1:
        devices = 'auto'
    else:
        devices = [int(x) for x in gpus.split(',')]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        benchmark=args.benchmark,
        logger=False
    )

    loguru_logger.info(f"Start testing!")
    # Lightning 2.x: verbose parameter removed from test()
    trainer.test(model, datamodule=data_module)
