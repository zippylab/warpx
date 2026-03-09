"""Input generation & validation utilities for WarpX."""

from .electrostatic_plasma import (  # noqa: F401
    ElectrostaticPlasmaSpec,
    generate_inputs_electrostatic_plasma,
)
from .electrostatic_plasma_validate import validate_electrostatic_plasma_spec  # noqa: F401
from .blocks import (  # noqa: F401
    AMRSpec,
    CollisionSpec,
    DiagSpec,
    DomainSpec,
    EBSpec,
    EMSolverSpec,
    ESSolverSpec,
    ExtBFieldSpec,
    HybridIonSpec,
    ImplicitSolverSpec,
    LaserSpec,
    OhmSolverSpec,
    ParticleBeamSpec,
    SolverSpec,
    SpeciesDefSpec,
    SpeciesSpec,
    _AMR_KEY_MAP,
    _emit_amr_block,
    suggest_cells,
    validate_amr,
)
from .electromagnetic_pic import (  # noqa: F401
    ElectromagneticPICSpec,
    generate_inputs_electromagnetic_pic,
)
from .electromagnetic_pic_validate import validate_electromagnetic_pic_spec  # noqa: F401
from .electrostatic_pic import (  # noqa: F401
    ElectrostaticPICSpec,
    generate_inputs_electrostatic_pic,
)
from .electrostatic_pic_validate import validate_electrostatic_pic_spec  # noqa: F401
from .hybrid_plasma import (  # noqa: F401
    HybridPlasmaSpec,
    generate_inputs_hybrid_plasma,
)
from .hybrid_plasma_validate import validate_hybrid_plasma_spec  # noqa: F401
from .ion_beam_instability import (  # noqa: F401
    IonBeamInstabilitySpec,
    generate_inputs_ion_beam_instability,
)
from .ion_beam_instability_validate import validate_ion_beam_instability_spec  # noqa: F401
from .magnetic_reconnection import (  # noqa: F401
    MagneticReconnectionSpec,
    generate_inputs_magnetic_reconnection,
)
from .magnetic_reconnection_validate import validate_magnetic_reconnection_spec  # noqa: F401
from .pwfa import PWFASpec, generate_inputs_pwfa  # noqa: F401
from .pwfa_validate import validate_pwfa_spec  # noqa: F401
from .laser_acceleration import (  # noqa: F401
    LaserAccelerationSpec,
    generate_picmi_laser_acceleration,
)
from .laser_acceleration_validate import (  # noqa: F401
    validate_laser_acceleration_spec,
)
from .native import (  # noqa: F401
    generate_inputs_uniform_plasma,
)
from .run import (  # noqa: F401
    dry_run_picmi_initialize_inputs,
    dry_run_native_max_step_zero,
)
from .spec import (  # noqa: F401
    Issue,
    Severity,
    UniformPlasmaSpec,
    ValidationReport,
)
from .validate import (  # noqa: F401
    validate_picmi_syntax,
    validate_uniform_plasma_spec,
)

# Backward compatible name
from .generate import generate_picmi_uniform_plasma  # noqa: F401
