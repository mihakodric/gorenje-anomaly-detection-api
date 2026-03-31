from utils.model import *
import argparse

parser = Parser()
predictor = HealthCheck()

input_parser = argparse.ArgumentParser()
input_parser.add_argument(
    "--filename",
    required=True,
    help="Path to the input JSON file containing cycle settings and results.",
)

args = input_parser.parse_args()

auid, cycle_settings, cycle_result = parser(args.filename)
json = predictor(cycle_settings, cycle_result, auid)

if __name__ == "__main__":
    print(json)