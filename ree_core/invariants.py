def check_residue_non_negative(residue_field):
    total = residue_field.total_mass()
    assert total >= 0.0, "Residue mass became negative!"

def check_residue_persistence(previous_mass, residue_field):
    current = residue_field.total_mass()
    assert current >= previous_mass, "Residue decreased unexpectedly!"