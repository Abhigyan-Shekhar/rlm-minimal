import os
import json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
import sys

# Add RLM to path
sys.path.insert(0, os.path.abspath('.'))
from rlm.rlm_repl import RLM_REPL
from dotenv import load_dotenv

load_dotenv()

def extract_answer(response):
    response = response.replace('*', '')
    # LongBench-v2 specific extraction
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            # Fallback for just finding A, B, C, D in the response if it's short
            if len(response.strip()) == 1 and response.strip() in "ABCD":
                return response.strip()
            return None

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(args.save_dir, "rlm_gemini_2_0_flash.jsonl")

    # Load dataset
    print("Loading LongBench-v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    data_all = [{"_id": item["_id"], "question": item["question"], 
                "choice_A": item["choice_A"], "choice_B": item["choice_B"], 
                "choice_C": item["choice_C"], "choice_D": item["choice_D"], 
                "answer": item["answer"], "context": item["context"]} for item in dataset]

    # Handle limit
    if args.limit > 0:
        data_all = data_all[:args.limit]

    # Cache handling
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            for line in f:
                try:
                    has_data[json.loads(line)["_id"]] = 0
                except:
                    continue
    
    fout = open(out_file, 'a', encoding='utf-8')
    
    # Initialize RLM
    print("Initializing RLM with Gemini 2.0 Flash...")
    rlm = RLM_REPL(
        provider="gemini",
        model="gemini-2.0-flash",
        recursive_model="gemini-2.0-flash",
        enable_logging=False,
        max_iterations=5, # Reduced for speed/cost during initial test
    )

    for item in tqdm(data_all):
        if item["_id"] in has_data:
            continue
            
        query = f"""Question: {item['question']}
Choices:
(A) {item['choice_A']}
(B) {item['choice_B']}
(C) {item['choice_C']}
(D) {item['choice_D']}

Please provide the correct answer in the format: "The correct answer is (X)" where X is A, B, C, or D."""

        try:
            # Use RLM to complete
            response = rlm.completion(context=item['context'], query=query)
            
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            
            # Save minimal context to save space
            item['context_snippet'] = item['context'][:500]
            del item['context']
            
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()
        except Exception as e:
            print(f"Error processing {item['_id']}: {e}")
            time.sleep(1)

    fout.close()
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="LongBench/results")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit number of samples to process")
    args = parser.parse_args()
    main(args)
