
def cache_init(num_steps, rel_l1_thresh=None, coefficients=None):
    '''
    Initialization for TeaCache.

    Args:
        num_steps: Total number of denoising steps.
        rel_l1_thresh: Threshold for accumulated relative L1 distance (lambda).
            Can be overridden by TEACACHE_REL_L1_THRESH env var.
        coefficients: 4th-order polynomial coefficients for rescaling rel_l1.
            Can be overridden by TEACACHE_COEFFICIENTS env var (comma-separated).

    Returns:
        (cache_dic, current) tuple.
    '''
    import os

    # Default coefficients from flux TeaCache
    default_coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]

    # Read from environment variables with fallback
    if rel_l1_thresh is None:
        rel_l1_thresh = float(os.environ.get('TEACACHE_REL_L1_THRESH', 0.6))

    if coefficients is None:
        coeff_str = os.environ.get('TEACACHE_COEFFICIENTS', None)
        if coeff_str is not None:
            coefficients = [float(x) for x in coeff_str.split(',')]
        else:
            coefficients = default_coefficients

    cache_dic = {}
    cache_dic['num_steps'] = num_steps
    cache_dic['rel_l1_thresh'] = rel_l1_thresh
    cache_dic['coefficients'] = coefficients

    current = {}
    current['cnt'] = 0
    current['accumulated_rel_l1_distance'] = 0
    current['previous_modulated_input'] = None
    current['previous_residual'] = None

    return cache_dic, current
