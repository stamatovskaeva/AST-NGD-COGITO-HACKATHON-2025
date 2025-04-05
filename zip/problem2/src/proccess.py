import cv2


def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess the image for the model.
    Args:
        img: Input image (numpy array).
        target_size: Target size for resizing the image.
    Returns:
        Preprocessed image (numpy array).
    """
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)