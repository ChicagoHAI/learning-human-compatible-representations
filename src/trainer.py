from omegaconf import OmegaConf as oc
import pytorch_lightning as pl
import torch, argparse
from pytorch_lightning.loggers import WandbLogger
import wandb
import time, os
import numpy as np
from pathlib import Path
import shutil

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default='configs/base.yaml', type=str, required=False)
    parser.add_argument("--model_config", default=None, type=str, required=True)
    parser.add_argument("--dataset_config", default=None, type=str, required=False)
    parser.add_argument("--triplet_config", default=None, type=str, required=False)
    parser.add_argument("--overwrite_config", default=None, type=str, required=False)

    ### overwrite args ###
    parser.add_argument("--wandb_project", default=None, type=str)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_name", default=None, type=str)

    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--train_batch_size", default=None, type=int)
    parser.add_argument("--lamda", default=None, type=float)
    parser.add_argument("--embed_dim", default=None, type=int)
    parser.add_argument("--max_epochs", default=None, type=int)
    
    parser.add_argument("--embeds_output_dir", default=None, type=str)
    parser.add_argument("--out_csv", default=None, type=str)

    parser.add_argument("--seed", default=None, type=int)
    return parser

def test_configs(configs):
    ''' checks for expected hyperparameters'''
    wandb_args = ["wandb_group", "wandb_mode", "wandb_project", "wandb_entity",  "wandb_name"]
    trainer_args = ["gpus", "seed",  "deterministic", "dataloader_num_workers",
        "do_train", "do_test", "checkpoint_callback", "enable_progress_bar", "log_every_n_steps"]
    training_args = ["max_epochs", "learning_rate","check_val_every_n_epoch",
        "train_batch_size", "triplet_batch_size"]
    dataset_args = ["train_dir", "valid_dir", "test_dir", "transform", "aug", "num_class", "syn"]
    triplet_args = ["train_triplets", "valid_triplets", "test_triplets", "filtered"]
    model_args = ["model", "embed_dim","pretrained", "lamda"]
    file_args = ["embeds_output_dir", "test_ckpt_path","out_csv"]
    ds_args = ["predicted_labels"]
    required_args = wandb_args + trainer_args + training_args + dataset_args + triplet_args + model_args + file_args

    if "syn" in configs:
        if configs["syn"]: 
            syn_args = ["syn", "train_synthetic", "valid_synthetic", "test_synthetic", "weights"]
            for arg in syn_args: assert(arg in configs)
            required_args += syn_args
            
    if set(configs) != set(required_args):
        missing_args = np.setdiff1d(required_args, configs)
        print("\n WARNING: Missing args:")
        print(missing_args)
        print("WARNING: Unrecognized args:")
        print(np.setdiff1d(configs, required_args))

        for key in missing_args: configs[key] = None

    return configs

def load_configs(args):
    '''' triplet_config > model_config > dataset_config > base_config '''
    base_config = oc.load(args.base_config)
    model_config = oc.load(args.model_config)
    dataset_config = oc.load(args.dataset_config) if args.dataset_config else {}
    triplet_config = oc.load(args.triplet_config) if args.triplet_config else {}
    overwrite_config = oc.load(args.overwrite_config) if args.overwrite_config else {}
    configs = oc.merge(base_config, dataset_config,  model_config, triplet_config)
    configs = test_configs(configs)
    args_override = {}
    args = vars(args)
    for hp in args:
        if 'config' not in hp and args[hp] is not None:
            args_override[hp] = args[hp]
            print("Args overwriting", hp, "with", args[hp])
    args_override = oc.create(args_override)
    configs = oc.merge(configs, overwrite_config, args_override)
    return configs

def load_configs_sweep(args):
    '''' triplet_config > model_config > dataset_config > base_config '''
    base_config = oc.load(args.base_config)
    model_config = oc.load(args.model_config)
    dataset_config = oc.load(args.dataset_config) if args.dataset_config else {}
    triplet_config = oc.load(args.triplet_config) if args.triplet_config else {}
    configs = oc.merge(base_config, dataset_config,  model_config, triplet_config)
    test_configs(configs)
    if args.learning_rate > 0:
        lr = oc.create({"learning_rate": args.learning_rate}) 
        configs = oc.merge(configs, lr)
    if args.train_batch_size > 0:
        bs = oc.create({"train_batch_size": args.train_batch_size}) 
        configs = oc.merge(configs, bs)
    seed = oc.create({"seed":args.seed})
    configs = oc.merge(configs, seed)
    return configs

def generic_train(model, args, monitor, profiler=None, num_sanity_val_steps=2,
                    early_stopping_callback=False, extra_callbacks=[], checkpoint_callback=None, logging_callback=None,  **extra_train_kwargs):
    output_dir = os.path.join("checkpoints", model.hparams.wandb_project)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args["wandb_name"] else args["wandb_name"]
    experiment = wandb.init(
        entity=args["wandb_entity"],
        project=args["wandb_project"],
        mode=args["wandb_mode"], 
        group=args["wandb_group"],
        name=wandb_name,
        config=args)

    logger = WandbLogger(project="imagenet_bm", experiment=experiment)

    train_params = {}
    train_params["max_epochs"] = args["max_epochs"]
    train_params["check_val_every_n_epoch"] = args["check_val_every_n_epoch"]
    train_params["enable_progress_bar"] = args["enable_progress_bar"]
    if args["gpus"] == -1 or args["gpus"] > 1:
        train_params["distributed_backend"] = "ddp"

    ckpt_path = os.path.join(output_dir, logger.version, "checkpoints")
    if args["checkpoint_callback"]:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_total_loss:.2f}", monitor=monitor, mode="min", save_last=True, save_top_k=3, verbose=True)
        train_params["callbacks"] = extra_callbacks + [checkpoint_callback]

    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=args["gpus"],
        weights_summary=None,
        logger=logger,
        profiler=profiler,
        num_sanity_val_steps=num_sanity_val_steps,
        **train_params)

    if args["do_train"]:
        model.train()
        trainer.fit(model)
        if args["checkpoint_callback"]:
            target_path = os.path.join(ckpt_path, 'best_model.ckpt')
            print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
            shutil.copy(checkpoint_callback.best_model_path, target_path)

    if args["do_test"]:
        model.eval()
        test_ckpt_path = args.test_ckpt_path if args.test_ckpt_path else 'best'
        trainer.test(model, ckpt_path=test_ckpt_path)
    
    if args["checkpoint_callback"]:
        ckpts = [f for f in os.listdir(ckpt_path)]
        for ckpt in ckpts:
            if ckpt != "best_model.ckpt":
                os.remove(os.path.join(ckpt_path, ckpt))

    return trainer

def do_test(model, args, ckpt_path):
    ''' generic_test '''
    wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args["wandb_name"] else args["wandb_name"]
    experiment = wandb.init(
        entity=args["wandb_entity"],
        project=args["wandb_project"],
        mode=args["wandb_mode"], 
        group=args["wandb_group"],
        name=wandb_name)

    logger = WandbLogger(experiment=experiment)

    train_params = {}
    train_params["max_epochs"] = args["max_epochs"]
    if args["gpus"] == -1 or args["gpus"] > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(
        gpus=args["gpus"],
        auto_select_gpus=True, ## true
        weights_summary=None,
        logger=logger,
        enable_progress_bar=False,
        check_val_every_n_epoch=1,
        **train_params)
        
    trainer.test(model, ckpt_path=ckpt_path)

    return trainer

def get_dataloader(dataset, batch_size, split, num_workers):
    drop_last = True if split == "train" else False
    shuffle = True if split == "train" else False
    batch_size = min(len(dataset), batch_size)
    batch_size = min(100,batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        num_workers=num_workers, drop_last=drop_last, shuffle=shuffle)
    return dataloader

def sample_dataset(dataset, index):
    input = torch.tensor(np.array([data[0].numpy() for data in dataset]))[index]
    label = torch.tensor(np.array([data[1] for data in dataset]))[index]
    dataset = torch.utils.data.TensorDataset(input, label)
    return dataset
