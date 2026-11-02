OPERATOR_LIBRARY = {
    "preprocess": {
        "gaussian": {
            "defaults": {"kernel": 3, "sigma": 1.0},
            "params": {
                "kernel": {"type": "int", "choices": [3, 5]},
                "sigma": {"type": "float", "min": 0.0, "max": 3.0},
            },
        },
        "median": {
            "defaults": {"kernel": 3},
            "params": {"kernel": {"type": "int", "choices": [3, 5]}},
        },
        "gamma": {
            "defaults": {"gamma": 1.4},
            "params": {"gamma": {"type": "float", "min": 0.2, "max": 3.0}},
        },
        "clahe": {
            "defaults": {"clip_limit": 2.0, "tile_grid": 8},
            "params": {
                "clip_limit": {"type": "float", "min": 0.1, "max": 8.0},
                "tile_grid": {"type": "int", "choices": [4, 8, 16]},
            },
        },
        "sharpen": {
            "defaults": {"strength": 0.5},
            "params": {"strength": {"type": "float", "min": 0.0, "max": 2.0}},
        },
    },
    "edge_detection": {
        "roberts": {
            "defaults": {"threshold": 55},
            "params": {"threshold": {"type": "int", "min": 0, "max": 255}},
        },
        "sobel": {
            "defaults": {"threshold": 80},
            "params": {"threshold": {"type": "int", "min": 0, "max": 255}},
        },
        "canny": {
            "defaults": {"low_threshold": 50, "high_threshold": 120},
            "params": {
                "low_threshold": {"type": "int", "min": 0, "max": 254},
                "high_threshold": {"type": "int", "min": 1, "max": 255},
            },
        },
        "hed": {"defaults": {}, "params": {}},
    },
    "postprocess": {
        "erosion": {
            "defaults": {"kernel": 3, "iterations": 1},
            "params": {
                "kernel": {"type": "int", "choices": [3, 5]},
                "iterations": {"type": "int", "min": 1, "max": 3},
            },
        },
        "dilation": {
            "defaults": {"kernel": 3, "iterations": 1},
            "params": {
                "kernel": {"type": "int", "choices": [3, 5]},
                "iterations": {"type": "int", "min": 1, "max": 3},
            },
        },
        "opening": {
            "defaults": {"kernel": 3, "iterations": 1},
            "params": {
                "kernel": {"type": "int", "choices": [3, 5]},
                "iterations": {"type": "int", "min": 1, "max": 3},
            },
        },
        "closing": {
            "defaults": {"kernel": 3, "iterations": 1},
            "params": {
                "kernel": {"type": "int", "choices": [3, 5]},
                "iterations": {"type": "int", "min": 1, "max": 3},
            },
        },
    },
}

STAGE_BY_OPERATOR = {
    op: stage
    for stage, ops in OPERATOR_LIBRARY.items()
    for op in ops
}
