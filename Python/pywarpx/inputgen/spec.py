from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    QUESTION = "question"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    issues: List[Issue] = field(default_factory=list)

    def add(self, severity: Severity, code: str, message: str, **details: Any) -> None:
        self.issues.append(Issue(severity=severity, code=code, message=message, details=details))

    def merge(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    @property
    def ok(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def questions(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.QUESTION]


@dataclass
class UniformPlasmaSpec:
    """A minimal, structured spec for a uniform plasma test problem."""

    # Geometry
    dim: int  # 1, 2, or 3
    number_of_cells: List[int]  # len == dim
    lower_bound: List[float]  # len == dim
    upper_bound: List[float]  # len == dim
    field_bc: List[str]  # len == dim

    # Time
    max_steps: int
    time_step_size: Optional[float] = None
    cfl: float = 0.99

    # Plasma
    density: float = 1.0e20  # m^-3
    temperature_eV: float = 1.0

    # Numerics
    particle_shape: int = 1
    current_deposition_algo: str = "direct"

    # Diagnostics
    diag_period: int = 10
    diag_fields: List[str] = field(default_factory=lambda: ["E", "B"])  # avoid J by default (not always available)
    diag_format: str = "plotfile"   # "plotfile" | "openpmd"
    write_species: bool = False

    # WarpX-specific optional knobs
    warpx_max_grid_size: Optional[int] = None

    # AMR settings (flat; kept here to avoid circular import with blocks.py)
    amr_max_level: int = 0
    amr_blocking_factor: int = 8

    # Metadata
    name: str = "uniform_plasma"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "number_of_cells": list(self.number_of_cells),
            "lower_bound": list(self.lower_bound),
            "upper_bound": list(self.upper_bound),
            "field_bc": list(self.field_bc),
            "max_steps": self.max_steps,
            "time_step_size": self.time_step_size,
            "cfl": self.cfl,
            "density": self.density,
            "temperature_eV": self.temperature_eV,
            "particle_shape": self.particle_shape,
            "current_deposition_algo": self.current_deposition_algo,
            "diag_period": self.diag_period,
            "diag_fields": list(self.diag_fields),
            "diag_format": self.diag_format,
            "write_species": self.write_species,
            "warpx_max_grid_size": self.warpx_max_grid_size,
            "amr_max_level": self.amr_max_level,
            "amr_blocking_factor": self.amr_blocking_factor,
        }
