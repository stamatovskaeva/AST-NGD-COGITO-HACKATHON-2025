import argparse
from src.model_training import run_evaluation, run_inference

def main():
    parser = argparse.ArgumentParser(description="Grocery Product Recognition Evaluation")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Path to a validation folder for inference mode")
    args = parser.parse_args()

    if args.val_dir:
        print("Running inference on validation directory:", args.val_dir)
        run_inference(args.val_dir)
    else:
        print("Running evaluation on test set (cropped_testing)...")
        run_evaluation()

if __name__ == "__main__":
    main()