# Problem 1: Product Recognition

**Can your AI distinguish a red paprika from a banana?** Or tell the difference between regular and sugar-free Red Bull cans that look nearly identical?

![Paprika Awareness Challenge](../images/paprika%20awareness.png)

The goal of this problem is to create a **computer vision model** that accurately predicts the product number (PLU/GTIN) based solely on image input. This is the foundation of an intelligent self-checkout system that can:

- **Verify** that the scanned product matches what's actually in the checkout area
- **Identify** products that might be difficult for customers (like produce without barcodes)
- **Reduce** loss by ensuring accurate product recognition

## Your Challenge

Develop a robust computer vision model that can correctly classify products across our dataset spanning **26 unique product categories** – from bananas and red apples to Kvikk Lunsj chocolate and Pepsi Max bottles.

## Technical Considerations

- **Dataset Structure**: As detailed in the data introduction notebook, you'll work with a dataset of product images organized by PLU/GTIN codes, with annotation files containing bounding box coordinates.
- **Model Architecture**: You're free to choose your preferred model architecture – CNN, YOLO, EfficientNet, ResNet, Vision Transformer, or any other approach.
- **Training Time**: Carefully consider computational requirements, as training or fine-tuning can take several hours.
- **Evaluation**: Your model will be evaluated on unseen test images based on classification accuracy.
- **Class Imbalance**: Note that some products have more training samples than others (as shown in the data statistics).

## Pro Tips

- Consider data augmentation to handle the varying angles and lighting conditions in a real store environment
- Transfer learning from pre-trained models can significantly reduce training time
- Pay special attention to similar-looking products that are challenging to distinguish

This is your chance to build a solution that could revolutionize retail technology and save millions in losses!
