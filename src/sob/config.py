import argparse
import yaml
import os
from dataclasses import dataclass, fields
from typing import Optional, Tuple, Literal, Union, get_type_hints, get_origin, get_args


@dataclass
class TrainingConfig:
    """Configuration class for training depth and pose estimation models.

    This dataclass stores all configuration parameters used for training, including
    model architecture, optimization, data, and loss settings.
    """

    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-4
    scheduler_step_size: int = 15
    run_name: str = "default"
    checkpoint_metric: Literal["AbsRel", "MAE"] = "AbsRel"
    load_run: Optional[str] = None
    load_best: bool = False
    project_path: str = ""
    no_compile: bool = False
    overfit: bool = False

    data_path: str = "data"
    pretrain_path: str = None
    dataset: Literal["kitti"] = "kitti"
    height: int = 192
    width: int = 640
    sources: Tuple = (-1, 1)
    num_scales: int = 4
    num_workers: int = 8
    no_inv: bool = False
    distribution: Literal["gaussian"] = "gaussian"
    components: int = 2
    alpha_entropy: float = 0.0
    alpha_smooth: float = 0.0
    sigma_entropy: float = 1e-3
    sigma_loss: float = 1.0
    smoothness: float = 1e-3
    smoothness_sigma: float = 1e-3
    alpha_loss: Literal["ce", "mse", "mae", "attn"] = "ce"
    encoder: Literal["convnext_base", "resnet18"] = "convnext_base"
    decoder: Literal["depth", "hr"] = "depth"
    sigma_type: Literal[
        "learned_dual", "learned_single", "fixed_depth_single", "fixed_depth_dual", "fixed_color", "none"
    ] = "learned_dual"
    num_layers: Optional[int] = None
    filter_min: Literal["default", "global", "alpha"] = "default"
    eigen_old: bool = False
    pc_error: bool = False
    finetune: bool = False

    config: Optional[str] = None

    def __post_init__(self):
        """Initialize paths after instance creation and validate choices.

        Expands environment variables in path strings and validates that attribute values
        match their Literal type constraints.
        """
        # Expand any environment variables in the path strings.
        self.project_path = os.path.expandvars(self.project_path)
        self.data_path = os.path.expandvars(self.data_path)
        if self.pretrain_path is not None:
            self.pretrain_path = os.path.expandvars(self.pretrain_path)
        if self.run_name == "default":
            print("[Warning] Run name is set to default")

    @classmethod
    def from_run(cls, run: str, project_path: str = "", **kwargs) -> "TrainingConfig":
        """Create a TrainingConfig from an existing run.

        Loads configuration from a YAML file in the specified run directory.

        Args:
            run (str): Name of the run to load configuration from.
            project_path (str, optional): Path to project directory.
                Defaults to "/home/aloception/.aloception/sob/".

        Returns:
            TrainingConfig: Configuration loaded from the run directory.
        """

        project_path = os.path.expandvars(project_path)

        config_path = os.path.join(project_path, run, "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # ignore args that are not in the config
        for key in tuple(config.keys()):
            if key not in cls.__annotations__:
                del config[key]

        if "enc_name" in config:
            config["encoder"] = config["enc_name"]
            del config["enc_name"]

        config["load_run"] = run
        config["run_name"] = run
        config["pretrain_path"] = None
        # change kwargs to config
        for key, value in kwargs.items():
            config[key] = value

        return cls(**config)

    @classmethod
    def from_args(cls, args=None) -> "TrainingConfig":
        """Create a TrainingConfig instance from command line arguments and/or config file.

        Returns:
            TrainingConfig: Configuration instance with parsed values
        """
        parser = argparse.ArgumentParser(description="Training configuration")

        # Add argument for config file
        parser.add_argument("--config", type=str, help="Path to config YAML file")

        # Get type hints for all fields
        type_hints = get_type_hints(cls)

        # Add arguments for all fields in the dataclass, excluding the config field
        for field in fields(cls):
            if field.name != "config":  # Skip config field
                field_type = type_hints[field.name]

                # Check if field has Literal type or Optional[Literal]
                choices = None
                if get_origin(field_type) is Literal:
                    choices = get_args(field_type)
                elif get_origin(field_type) is Union:
                    for arg in get_args(field_type):
                        if get_origin(arg) is Literal:
                            choices = get_args(arg)
                            break

                if choices:
                    # For fields with Literal type, add choices to the argument
                    parser.add_argument(
                        f"--{field.name}",
                        type=str,  # Use str type to handle all literal values
                        choices=choices,
                        default=field.default,
                        help=f'{field.name.replace("_", " ").title()} (choices: {", ".join(map(str, choices))})',
                    )
                elif field_type == bool:
                    parser.add_argument(
                        f"--{field.name}",
                        action="store_true",
                        default=field.default,
                        help=f'{field.name.replace("_", " ").title()}',
                    )
                elif field.name == "sources":
                    # Special handling for sources tuple
                    parser.add_argument(
                        "--sources",
                        nargs="+",  # Accept one or more arguments
                        type=int,
                        default=field.default,
                        help="Source types as numerical offsets (e.g., '-1 1' or 'stereo' or '-2 -1 1 2')",
                    )
                else:
                    # Use appropriate type for the argument
                    arg_type = field_type
                    if get_origin(field_type) is Union:
                        # For Optional types, use the non-None type
                        non_none_types = [t for t in get_args(field_type) if t is not type(None)]
                        if non_none_types:
                            arg_type = non_none_types[0]

                    parser.add_argument(
                        f"--{field.name}",
                        type=arg_type,
                        default=field.default,
                        help=f'{field.name.replace("_", " ").title()}',
                    )

        args = parser.parse_args(args)

        # Process sources
        if isinstance(args.sources, list):
            args.sources = tuple(args.sources)

        # If config file is provided, update args with config file values
        if args.config:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
                # Update args with config file values, but don't override command line args
                arg_dict = vars(args)
                for key, value in config.items():
                    if arg_dict[key] == parser.get_default(key):
                        arg_dict[key] = value

        return cls(**vars(args))

    @property
    def HW(self):
        """Get height and width as a tuple.

        Returns:
            tuple: (height, width) dimensions.
        """
        return self.height, self.width

    @property
    def run_path(self):
        """Get the path for the current run.

        Returns:
            str: Path to the current run directory.
        """
        return os.path.join(self.project_path, self.run_name)

    def to_dict(self):
        """Convert configuration to dictionary.

        Converts all configuration parameters to a dictionary, excluding None values
        and the config field.

        Returns:
            dict: Dictionary of configuration parameters.
        """
        return {k: v for k, v in vars(self).items() if v is not None and k != "config"}

    def __str__(self) -> str:
        """Get string representation of the configuration.

        Returns:
            str: String representation of the configuration dictionary.
        """
        return str(self.to_dict())

    def save(self) -> None:
        """Save the configuration to a YAML file."""
        # Create the run directory if it doesn't exist
        os.makedirs(self.run_path, exist_ok=True)

        filepath = os.path.join(self.run_path, "config.yaml")
        # Convert dataclass to dictionary, excluding None values and config field
        config_dict = self.to_dict()
        # Write to YAML file
        with open(filepath, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)


if __name__ == "__main__":
    config = TrainingConfig()
    print(config.to_dict())
