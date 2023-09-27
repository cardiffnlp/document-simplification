import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from plan_simp.data.roberta import RobertaDataModule
from plan_simp.models.classifier import RobertaClfFinetuner


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()

    # add model specific args to parser
    parser = RobertaClfFinetuner.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare data module and finetuner class
    if args.checkpoint is None:
        model = RobertaClfFinetuner(add_context=args.add_context, params=args)
    else:
        model = RobertaClfFinetuner.load_from_checkpoint(
                    args.checkpoint, add_context=args.add_context, params=args,
                    strict=False)

    dm = RobertaDataModule(model.tokenizer, params=args)

    if args.name is None:
        # use default logger settings (for hparam sweeps)
        wandb_logger = WandbLogger()
        checkpoint_callback=None
    else:
        # prepare logger
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, save_dir=args.save_dir, id=args.wandb_id)

        # checkpoint callback
    mode = "max" if args.ckpt_metric.split("_")[-1] == "f1" else "min"
    checkpoint_callback = ModelCheckpoint(monitor=args.ckpt_metric, mode=mode, save_top_k=1, save_last=True)
    
    # early_stop_callback = EarlyStopping(monitor="val_macro_f1", min_delta=0.000, patience=3, verbose=False, mode="max")
    # print(checkpoint_callback)
    trainer = pl.Trainer.from_argparse_args(
        args,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        accelerator="gpu",
        strategy="ddp",
        #plugins=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
        precision=16,)

    trainer.fit(model, dm)
