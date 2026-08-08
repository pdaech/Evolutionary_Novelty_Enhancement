import argparse


def args():

    parser = argparse.ArgumentParser(description="Runs Evolutionary Novelty Pipeline")

    parser.add_argument("--experiment_id", type=str, help="Id of the experiment")
    parser.add_argument("--evaluator_index", type=int, help="Index of the experiment")

    parser.add_argument("--pcas", type=str, help="prompt of the pcas")

    parser.add_argument("--ev", type=str, help="evaluator_funciton")
    parser.add_argument("--prompt", type=str, help="prompt for this experiment")
    parser.add_argument("--new_prompt", type=str, help="prompt for this experiment")
    args = parser.parse_args()

    return args
