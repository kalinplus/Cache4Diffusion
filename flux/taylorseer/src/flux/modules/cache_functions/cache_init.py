import os


def cache_init(**kwargs):
    """
    Initialization for cache.
    """
    use_smoothing = os.environ.get("USE_SMOOTHING", "False").lower() in ("true", "1", "yes")
    use_hybrid_smoothing = os.environ.get("USE_HYBRID_SMOOTHING", "False").lower() in ("true", "1", "yes")
    smoothing_method = os.environ.get("SMOOTHING_METHOD", "exponential")
    smoothing_alpha = float(os.environ.get("SMOOTHING_ALPHA", "0.8"))

    cache = {}
    cache[-1] = {}
    cache[-2] = {}

    cache[-1]["double_stream"] = {}
    cache[-2]["double_stream"] = {}
    cache[-1]["single_stream"] = {}
    cache[-2]["single_stream"] = {}

    for i in range(19):
        cache[-1]["double_stream"][i] = {}
        cache[-2]["double_stream"][i] = {}

    for i in range(38):
        cache[-1]["single_stream"][i] = {}
        cache[-2]["single_stream"][i] = {}

    cache_dic = {}
    cache_dic["cache"] = cache
    cache_dic["num_steps"] = kwargs["num_steps"]
    cache_dic["test_FLOPs"] = kwargs["test_FLOPs"]
    cache_dic["monitor_gpu_usage"] = kwargs["monitor_gpu_usage"]
    cache_dic["interval"] = kwargs["interval"]
    cache_dic["max_order"] = kwargs["max_order"]
    cache_dic["first_enhance"] = kwargs["first_enhance"]
    cache_dic["taylor_cache"] = True
    cache_dic["use_smoothing"] = use_smoothing
    cache_dic["use_hybrid_smoothing"] = use_hybrid_smoothing
    cache_dic["smoothing_method"] = smoothing_method
    cache_dic["smoothing_alpha"] = smoothing_alpha

    current = {}
    current["step"] = 0
    current["activated_steps"] = [0]
    current["cache_counter"] = 0

    return cache_dic, current
