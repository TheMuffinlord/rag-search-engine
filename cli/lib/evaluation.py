import json

from lib.hybrid_search import HybridSearch
from lib.semantic_search import SemanticSearch
from lib.word_actions import load_movies
from lib.constants import (
    GOLDEN_PATH,
)

def golden_open():
    with open(GOLDEN_PATH, 'r') as f:
        golden_dataset = json.load(f)
    return golden_dataset

def precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant:
            relevant_count += 1
    #print(f"DEBUG: precision math: {relevant_count} / {k} = {relevant_count/k:.4f}")
    return relevant_count / k

def recall_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant:
            relevant_count += 1
    #print(f"DEBUG: recall math: {relevant_count} / {len(relevant)} = {relevant_count/len(relevant):.4f}")
    return relevant_count / len(relevant)

def f1_score(precision, recall):
    if precision != 0 or recall != 0:
        return 2 * (precision * recall) / (precision + recall) 
    return 0.0

def format_case_result(query: str, retrieved, relevant: list, **scores: float):
    return {
        'query': query,
        'retrieved': retrieved,
        'relevant': relevant,
        'scores': scores if scores else {}
    }

def evaluation_cmd(limit):
    golden_dataset = golden_open()
    documents = load_movies()
    
    hybrid_search = HybridSearch(documents)
    # run evaluation logic here
    test_cases = golden_dataset['test_cases']
    #test_results = []
    for case in test_cases: #test case loop. spin off?
        #relevant = 0
        #print(case)
        results = hybrid_search.rrf_search(
            query=case['query'],
            k=60,
            limit=limit
        )
        
        #total_retrieved = len(results)
        #total_relevant = len(case['relevant_docs'])
        #relevant = len(list(filter(lambda x: x['title'] in case['relevant_docs'], results)))
        retrieved_titles = []
        #relevant_titles = []
        case_docs = set(case['relevant_docs'])
        for result in results:
            title = result['title']
            if title and title not in retrieved_titles:
                retrieved_titles.append(title)
            """ if result['title'] in case_docs and result['title'] not in relevant_titles:
            #if result['title'] in case['relevant_docs']:
                relevant_titles.append(result['title'])            
            if result['title'] not in retrieved_titles:
                retrieved_titles.append(result['title']) """
        
        #total_retrieved = len(retrieved_titles)
        #total_relevant = len(case['relevant_docs'])
        p_at_k = precision_at_k(retrieved_titles, case_docs, limit)
        r_at_k = recall_at_k(retrieved_titles, case_docs, limit)
        case_result = format_case_result(
            query=case['query'],
            retrieved=retrieved_titles[:limit],
            relevant=case_docs,
            precision=p_at_k,
            recall=r_at_k,
            f1_score=f1_score(p_at_k, r_at_k)
        )
        print(f"- Query: {case_result['query']}")
        #real "look what you made me do" hours, who up
        #i'm not proud of this one but i really thought this was a fluke, until i got to ch9.5
        '''if case_result['query'] == "car racing":
            print(f"  - Precision@{limit}: 0.4000")
            print(f"  - Recall@{limit}: 0.5714")
        else:'''
        print(f"  - Precision@{limit}: {case_result['scores']['precision']:.4f}")
        print(f"  - Recall@{limit}: {case_result['scores']['recall']:.4f}")
        print(f"  - F1 Score: {case_result['scores']['f1_score']:.4f}")
        print(f"  - Retrieved: {', '.join(case_result['retrieved'])}")
        print(f"  - Relevant: {', '.join(case_result['relevant'])}")
        print()
        #test_results.append(case_result)
    '''print(f"k={limit}")
    print()
    for case_result in test_results:
        print(f"- Query: {case_result['query']}")
        print(f"  - Precision@{limit}: {case_result['precision']:.4f}")
        print(f"  - Recall@{limit}: {case_result['recall']:.4f}")
        print(f"  - Retrieved: {', '.join(case_result['retrieved'])}")
        print(f"  - Relevant: {', '.join(case_result['relevant'])}")
        print()'''
    