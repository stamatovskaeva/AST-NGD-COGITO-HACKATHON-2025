# Problem 2: Receipt Generation from Video Analysis

**Can your system track an entire shopping journey?** From the moment a customer starts scanning items to the final receipt, your AI needs to follow every product.

![Video Frame Sample](../../images/video%20frame%2010%20and%2030.png)

The goal of this problem is to create a **video analysis system** that can produce a complete and accurate receipt by analyzing footage from the self-checkout cameras. This represents the ultimate test for a real-world deployment.

## Your Challenge

Develop a robust system that can:

- **Identify** each product as it appears into the scanning area in the video frame at what time
- **Count** multiple instances of the same product
- **Generate** a receipt of "scanned" products

## Technical Considerations

- **Video Analysis**: You'll work with footage from checkout cameras showing customers scanning various products
- **Temporal Analysis**: Your system needs to understand the shopping sequence, not just identify individual frames
- **Frame Selection**: As shown in the sample frames at 10 and 30 seconds, products appear differently throughout the scanning process
- **Receipt Matching**: Your generated receipt will be compared against the actual transaction data
- **Speed vs. Accuracy**: Balance processing speed with high accuracy for a practical solution

## Evaluation Criteria

Your solution will be scored primarily on:

- **Product Number Accuracy**: Correctly identifying each PLU/GTIN code, on the video running in real time (50%)
- **Quantity Accuracy**: Correctly counting multiple instances of the same product (25%)
- **Theft Detection**: **Extra points** for identifying products that appear in the video but were never scanned by the customer (25%)

For evaluation, we have the **true "checkout bong"** (receipt) from each transaction, which will be compared to:

1. The list of products your computer vision system detected
2. What was actually scanned in real life by the customer

## Pro Tips

- Consider using tracking algorithms to follow products across multiple frames
- Implement confidence thresholds to avoid false positives
- Frame differencing techniques can help identify when new products enter the scene
- Ensemble methods combining your Problem 1 model with temporal analysis often yield better results

This challenge represents the complete self-checkout monitoring system that could save retailers millions while improving customer experience!
