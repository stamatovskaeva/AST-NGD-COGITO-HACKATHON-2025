# Problem 1: Submission Guidelines

## What to Submit

For the Product Recognition challenge, your solution must deliver a computer vision model that accurately classifies grocery products from images.

### Key Deliverables

1. **Classification Model**: A trained model that can identify products from our dataset spanning 26 unique product categories
2. **Inference Code**: Code that can take a new image as input and output the predicted product number (PLU/GTIN)
3. **Documentation**: Clear instructions on how to use your model

## Code Structure

Your submission must adhere to the following structure:

```bash
DELIVERY
└── PROBLEM1
    ├── README.md         # Instructions for running your solution
    ├── main.py           # Entry point for your solution
    ├── requirements.txt  # List of required packages
    ├── model/            # Directory containing your trained model (if applicable)
    └── src/              # Additional source code
```

## Technical Requirements

1. **Input Handling**: Your `main.py` must accept a path to a validation folder that has the same structure as the training data
2. **Validation Structure**: The validation folder will have the same hierarchical structure as the dataset you've been provided
3. **Output Format**: Your solution must output prediction score for each of the items

4. **Command Line Interface**: Your solution should be runnable with:

   ```bash
   python main.py --val_dir /path/to/val_folder
   ```

5. **Dependencies**: All required packages must be listed in `requirements.txt` (including visualization libraries like matplotlib)

**IMPORTANT**: The only modification that should be required to run your code is changing the path to the validation folder. Your code should work on any similarly structured data without additional changes and should display the accuracy plot automatically.

## Evaluation Criteria

Your solution will be evaluated based on:

- **Classification Accuracy** (70%): How accurately your model predicts the correct PLU/GTIN codes, which should be clearly visualized in your accuracy plot
- **Speed** (15%): How quickly your model can process images
- **Code Quality** (15%): Organization, documentation, and efficiency of your solution

## README Requirements

Your README.md should include:

1. A brief description of your approach
2. Instructions for setting up and running your code
3. Any model architecture information or pretrained models used
4. Any data preprocessing or augmentation techniques employed
5. Known limitations or areas for improvement

Remember: The jury will execute only your `main.py` file, so ensure all necessary functions are properly imported and called from there.
