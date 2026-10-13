import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.consistency import InconsistencyDetector

def main():
    parser = argparse.ArgumentParser(description="Check text for semantic inconsistencies.")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    
    args = parser.parse_args()
    
    text = ""
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.text:
        text = args.text
    else:
        print("Please provide --text or --file")
        return

    print("Loading model...")
    detector = InconsistencyDetector()
    print("Analyzing...")
    result = detector.detect_inconsistency(text)
    
    print("\n--- Inconsistency Report ---")
    print(f"Overall Inconsistency Score: {result['inconsistency_score']:.2f}")
    
    if result['details']:
        print(f"\nFound {len(result['details'])} potential inconsistencies:")
        for i, det in enumerate(result['details']):
            print(f"\n{i+1}. Low Similarity ({det['similarity']:.2f}) between:")
            print(f"   Para A: {det['para_a']}")
            print(f"   Para B: {det['para_b']}")
    else:
        print("\nNo significant inconsistencies found.")

if __name__ == "__main__":
    main()
