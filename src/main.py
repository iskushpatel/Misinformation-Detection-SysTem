# import argparse
# from ingest import ingest_data
# from search import EvidenceRetriever
# from explain import generate_explanation
# def main():
#     parser = argparse.ArgumentParser(description="FactChk CLI Demo")
#     parser.add_argument("--mode", choices=["ingest", "check"], required=True, help="Mode: 'ingest' to load data, 'check' to verify a claim.")
#     parser.add_argument("--claim", type=str, help="The claim text to verify (required for check mode).")
    
#     args = parser.parse_args()

#     if args.mode == "ingest":
#         print("Initializing Knowledge Base...")
#         ingest_data()
        
#     elif args.mode == "check":
#         if not args.claim:
#             print("Error: Please provide a claim using --claim 'text'")
#             return
#         retriever = EvidenceRetriever()
#         results = retriever.search(args.claim)
#         report = generate_explanation(args.claim, results)
#         print(report)

# if __name__ == "__main__":
#     main()
import argparse
from search import EvidenceRetriever
from explain import generate_explanation


def main():
    parser = argparse.ArgumentParser(description="VeriVector CLI Demo")
    parser.add_argument("--claim", type=str, required=True)

    args = parser.parse_args()

    retriever = EvidenceRetriever()
    results = retriever.search(args.claim)
    report = generate_explanation(args.claim, results)

    print(report)


if __name__ == "__main__":
    main()
