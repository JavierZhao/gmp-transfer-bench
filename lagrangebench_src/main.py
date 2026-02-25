import os

from omegaconf import DictConfig, OmegaConf


def ensure_requested_gpu_available(gpu_id: int) -> None:
    """Fail fast when a GPU run is requested but JAX only sees CPU devices."""
    if gpu_id == -1:
        return

    import jax

    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if gpu_devices:
        print(f"JAX GPU devices detected: {gpu_devices}")
        return

    jax_platforms = os.environ.get("JAX_PLATFORMS", "<unset>")
    raise RuntimeError(
        "A GPU run was requested (gpu={gpu_id}), but JAX reported no GPU devices. "
        "Detected JAX devices: {devices}. "
        "JAX_PLATFORMS={jax_platforms}. "
        "This usually means the environment has CPU-only JAX wheels. "
        "Install GPU-enabled JAX (for example: `pip install -U \"jax[cuda12]==0.4.29\"`) "
        "and ensure the CUDA plugin packages are installed."
        .format(gpu_id=gpu_id, devices=devices, jax_platforms=jax_platforms)
    )


def check_subset(superset, subset, full_key=""):
    """Check that the keys of 'subset' are a subset of 'superset'."""
    for k, v in subset.items():
        key = full_key + k
        if isinstance(v, dict):
            check_subset(superset[k], v, key + ".")
        else:
            msg = f"cli_args must be a subset of the defaults. Wrong cli key: '{key}'"
            assert k in superset, msg


def load_embedded_configs(config_path: str, cli_args: DictConfig) -> DictConfig:
    """Loads all 'extends' embedded configs and merge them with the cli overwrites."""

    cfgs = [OmegaConf.load(config_path)]
    while "extends" in cfgs[0]:
        extends_path = cfgs[0]["extends"]
        del cfgs[0]["extends"]

        # go to parents configs until the defaults are reached
        if extends_path != "LAGRANGEBENCH_DEFAULTS":
            cfgs = [OmegaConf.load(extends_path)] + cfgs
        else:
            from lagrangebench.defaults import defaults

            cfgs = [defaults] + cfgs

            # assert that the cli_args are a subset of the defaults if inheritance from
            # defaults is used.
            check_subset(cfgs[0], cli_args)

            break

    # merge all embedded configs and give highest priority to cli_args
    cfg = OmegaConf.merge(*cfgs, cli_args)
    return cfg


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    assert ("config" in cli_args) != (
        "load_ckp" in cli_args
    ), "You must specify one of 'config' or 'load_ckp'."

    if "config" in cli_args:  # start from config.yaml
        config_path = cli_args.config
    elif "load_ckp" in cli_args:  # start from a checkpoint
        config_path = os.path.join(cli_args.load_ckp, "config.yaml")

    # values that need to be specified before importing jax
    cli_args.gpu = cli_args.get("gpu", -1)
    cli_args.xla_mem_fraction = cli_args.get("xla_mem_fraction", 0.75)

    # specify cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    if cli_args.gpu == -1:
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cli_args.xla_mem_fraction)

    ensure_requested_gpu_available(cli_args.gpu)

    # The following line makes the code deterministic on GPUs, but also extremely slow.
    # os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

    cfg = load_embedded_configs(config_path, cli_args)

    print("#" * 79, "\nStarting a LagrangeBench run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    from lagrangebench.runner import train_or_infer

    train_or_infer(cfg)
