from omegaconf import OmegaConf
import os

def get_cfg_from_args(args):
    args_dict = vars(args)
    args_cfg = OmegaConf.create(args_dict)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, args_cfg)
    return cfg

def write_config(cfg, output_dir, name="config.yaml"):
    # logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg_name = os.path.basename(args.config_file)
    write_config(cfg, args.output_dir, name=cfg_name)
    return cfg