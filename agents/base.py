"""Base agent class for design editing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import json
from lib.types import Spec


class Agent(ABC):
    """
    Base class for design editing agents.

    Agents take a multi-layer design spec and produce a modified version
    based on a text instruction.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the agent.

        Args:
            verbose: If True, print debug information during execution
        """
        self.verbose = verbose

    def load_spec(self, spec_path: Path) -> Spec:
        """
        Load a design spec from a JSON file.

        Args:
            spec_path: Path to the spec.json file

        Returns:
            Spec: Parsed design specification
        """
        spec_path = Path(spec_path)
        with open(spec_path, 'r') as f:
            spec_data = json.load(f)
        return Spec(**spec_data)

    def save_spec(self, spec: Spec, output_path: Path) -> Path:
        """
        Save a design spec to a JSON file.

        Args:
            spec: The design specification to save
            output_path: Path where the spec should be saved

        Returns:
            Path: The output path where the spec was saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for serialization
        spec_dict = spec.model_dump()

        with open(output_path, 'w') as f:
            json.dump(spec_dict, f, indent=2)

        return output_path

    @abstractmethod
    def edit(self, spec_path: Path, instruction: str, output_path: Path) -> Path:
        """
        Edit a design based on a text instruction.

        Args:
            spec_path: Path to the input design spec (spec.json)
            instruction: Natural language instruction for the edit
            output_path: Path where the edited spec should be saved

        Returns:
            Path: The output path where the edited spec was saved

        Example:
            >>> agent = SomeAgent()
            >>> agent.edit(
            ...     spec_path="datasets/canva_specs/design1/spec.json",
            ...     instruction="Make the title text larger and blue",
            ...     output_path="outputs/design1_edited/spec.json"
            ... )
        """
        pass

    def log(self, message: str):
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")
