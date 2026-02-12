from dataclasses import dataclass

@dataclass
class MechanismFlags:
    use_explicit_M_term: bool = True
    use_residue: bool = True
    use_offline_integration: bool = True