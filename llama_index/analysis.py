import json
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Analyze evaluation results.')
parser.add_argument('input_file', type=str, help='The JSON file containing the evaluation results.')
args = parser.parse_args()


def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_results(results):
    df = pd.DataFrame(results)
    
    # Value counts of raw scores
    raw_score_counts = df['rag_score'].value_counts().sort_index()
    print("Raw Score Counts:")
    print(raw_score_counts)
    
    # Normalize scores to a 0-1 scale
    df['normalized_score'] = df['rag_score'] / 5.0
    
    # Value counts of normalized scores
    normalized_score_counts = df['normalized_score'].value_counts().sort_index()
    print("\nNormalized Score Counts:")
    print(normalized_score_counts)
    
    # Calculate normalized accuracy
    normalized_accuracy = df['normalized_score'].mean()
    print(f"\nNormalized Accuracy: {normalized_accuracy:.2f}")

    # Save results to JSON file
    output_file = args.input_file.replace('.json', '_analysis.json')
    df.to_json(output_file, orient='records', lines=True)
    print(f"\nAnalysis results saved to {output_file}")


if __name__ == "__main__":
    results = load_results(args.input_file)
    analyze_results(results)