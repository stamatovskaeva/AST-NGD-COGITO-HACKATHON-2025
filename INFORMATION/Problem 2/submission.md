# Problem 2: Submission Guidelines

## What to Submit

For the Receipt Generation from Video challenge, your solution must deliver a system that can analyze checkout footage and produce an accurate digital receipt.

### Key Deliverables

1. **Video Analysis System**: Code that processes checkout videos
2. **Receipt Generation**: Functionality that creates an itemized receipt based on detected products
3. **Theft Detection**: (Bonus) Capability to identify unscanned items
4. **Documentation**: Clear instructions on how to use your system

## Code Structure

Your submission must adhere to the following structure:

```bash
DELIVERY
└── PROBLEM2
    ├── README.md         # Instructions for running your solution
    ├── main.py           # Entry point for your solution
    ├── requirements.txt  # List of required packages
    ├── model/            # Directory containing your trained model (if applicable)
    └── src/              # Additional source code
```

## Technical Requirements

1. **Input Handling**: Your `main.py` must accept a path to any .mp4 video file for analysis
2. **Video Processing**: Your system should play the video while simultaneously analyzing it
3. **Output Format**: Your solution must output a receipt as a CSV file named `run.csv` with the following columns:

   ```
   timestamp,item
   ```

   Where:

   - `timestamp` is the time in the video (in seconds)
   - `item` is the PLU/GTIN code of the detected product

4. **Command Line Interface**: Your solution should be runnable with:

   ```bash
   python main.py --video /path/to/video.mp4
   ```

5. **Dependencies**: All required packages must be listed in `requirements.txt`

**IMPORTANT**: We will test your system on similar, but not identical videos to what you've been provided. Your solution should be robust enough to handle variations in lighting, camera angle, products, and scanning patterns.

## Evaluation Criteria

Your solution will be evaluated based on:

- **Product Identification** (50%): How accurately your system identifies products in the video
- **Quantity Accuracy** (25%): How correctly your system counts multiple instances of the same product
- **Theft Detection** (25%): How effectively your system identifies products that appear in the video but weren't scanned

## README Requirements

Your README.md should include:

1. A description of your approach to video analysis
2. How your system handles temporal information and product tracking
3. Any techniques used for distinguishing between similar products
4. Your approach to detecting unscanned items (theft detection)
5. Instructions for setting up and running your code
6. Known limitations or areas for improvement

## Performance Notes

Since video processing can be resource-intensive, please include:

1. Hardware recommendations for optimal performance
2. Expected processing time per minute of video
3. Any options for trading off accuracy versus speed

Remember that your solution will be compared against the actual "checkout bong" (receipt) for evaluation, so focus on accurate identification above all else.
