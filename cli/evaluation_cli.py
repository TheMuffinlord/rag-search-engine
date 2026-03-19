import argparse, json
from lib.evaluation import evaluation_cmd

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit
    evaluation_cmd(limit)
    



if __name__ == "__main__":
    main()