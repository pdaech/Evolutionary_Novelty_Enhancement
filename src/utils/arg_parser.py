import argparse

from src.model_revisions import DEFAULT_SDXL_REVISION, full_model_revision


def args():

    parser = argparse.ArgumentParser(description="Runs Evolutionary Novelty Pipeline")

    parser.add_argument(
        "--experiment_id", type=str, required=True, help="Experiment id"
    )
    parser.add_argument("--evaluator_index", type=int, help="Index of the experiment")

    parser.add_argument("--pcas", type=str, help="prompt of the pcas")

    parser.add_argument(
        "--evaluator",
        "--ev",
        dest="evaluator",
        default="novelty",
        help="Fitness evaluator: novelty or gemma-creativity",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt")
    parser.add_argument("--new_prompt", type=str, help="prompt for this experiment")
    parser.add_argument(
        "--directory",
        type=str,
        default="0_results/simulations",
        help="Result path relative to BASE_PATH",
    )
    parser.add_argument(
        "--id", type=str, default="experiment", help="Experiment id prefix"
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_generations", type=int, default=30)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument(
        "--sdxl_revision",
        type=full_model_revision,
        default=DEFAULT_SDXL_REVISION,
        help="Full SDXL commit hash; defaults to the successful smoke-test snapshot",
    )
    parser.add_argument(
        "--gemma_model",
        type=str,
        default="google/gemma-4-26B-A4B-it",
    )
    parser.add_argument(
        "--gemma_revision",
        type=str,
        default=None,
        help=(
            "Full Hugging Face commit hash; defaults to the revision validated "
            "against the human ratings"
        ),
    )
    parser.add_argument("--gemma_max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    return args
