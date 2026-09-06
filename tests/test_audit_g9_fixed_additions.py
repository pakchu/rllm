from training import optimize_g9_plus_added_alphas as j


def test_fixed_sensitivity_cells_were_in_original_grid():
    _,names=j.allocation_grid()
    cells=[n for n in names if n.startswith('g9x')]
    assert len(cells)==27
    assert 'g9x0.5_macro_flow1.0' in cells
    assert 'g9x1.0_macro_flow0.5' in cells
