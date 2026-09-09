import argparse

DEFAULT_SDXL_NUM_INFERENCE_STEPS = 50
DEFAULT_SDXL_GUIDANCE_SCALE = 7.5
EXPECTED_SDXL_SCHEDULER_CLASS = "EulerDiscreteScheduler"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed
