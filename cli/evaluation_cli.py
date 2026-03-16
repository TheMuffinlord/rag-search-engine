import argparse, json

from lib.hybrid_search import HybridSearch
from lib.word_actions import load_movies, format_search_result
from lib.constants import (
    GOLDEN_PATH,
)

def golden_open():
    with open(GOLDEN_PATH, 'r') as f:
        golden_dataset = json.load(f)
    return golden_dataset

def format_case_result(query: str, precision: float, retrieved: list, relevant: list):
    return {
        'query': query,
        'precision': precision,
        'retrieved': retrieved,
        'relevant': relevant
    }

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit
    golden_dataset = golden_open()
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    # run evaluation logic here
    test_cases = golden_dataset['test_cases']
    test_results = []
    for case in test_cases: #test case loop. spin off?
        relevant = 0
        #print(case)
        results = hybrid_search.rrf_search(
            query=case['query'],
            k=60,
            limit=limit
        )
        total_retrieved = len(results)
        #relevant = len(list(filter(lambda x: x['title'] in case['relevant_docs'], results)))
        retrieved_titles = []
        relevant_titles = []
        for result in results:
            if result['title'] in case['relevant_docs']:
                relevant += 1
                relevant_titles.append(result['title'])
            retrieved_titles.append(result['title'])

        case_result = format_case_result(
            query=case['query'],
            precision=relevant/total_retrieved,
            retrieved=retrieved_titles,
            relevant=relevant_titles
        )
        test_results.append(case_result)
    print(f"k={limit}")
    print()
    for case_result in test_results:
        print(f"- Query: {case_result['query']}")
        print(f"  - Precision@{limit}: {case_result['precision']:.4f}")
        print(f"  - Retrieved: {', '.join(case_result['retrieved'])}")
        print(f"  - Relevant: {', '.join(case_result['relevant'])}")
        print()
    




if __name__ == "__main__":
    main()