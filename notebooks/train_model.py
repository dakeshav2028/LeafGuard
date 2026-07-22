import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2
from keras.utils import image_dataset_from_directory

def main():
    # Define directory paths
    DATA_DIR = "c:/MY_PROJECTS/Image_Classification_web_app/data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    MODEL_PATH = "c:/MY_PROJECTS/Image_Classification_web_app/models/plant_disease_model.keras"
    EVAL_DIR = "c:/MY_PROJECTS/Image_Classification_web_app/evaluation"
    
    # Ensure outputs folders exist
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Image size expected by MobileNetV2 is 224x224
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # We train for 10 epochs. Since we use transfer learning (frozen backbone), 
    # the number of trainable weights is very small, and we will converge rapidly
    # while keeping CPU training times to ~1-2 minutes.
    EPOCHS = 10
    
    print("\n--- Step 1: Loading Datasets ---")
    # Load training, validation, and test datasets
    # label_mode='categorical' tells Keras to return labels as one-hot encoded vectors (e.g., [1, 0, 0])
    train_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )
    
    val_dataset = image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    test_dataset = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # Extract class names from the dataset labels
    class_names = train_dataset.class_names
    print(f"Detected Classes: {class_names}")
    
    # Save classes to a file for reference in the FastAPI backend
    with open(os.path.join(EVAL_DIR, "classes.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")
    
    print("\n--- Step 2: Defining Data Augmentation ---")
    # Data augmentation helps regularize the model and prevent overfitting
    # by generating synthetic training images with slight modifications.
    # Note: Keras augmentation layers are active ONLY during training (`model.fit`).
    # During evaluation/inference (`model.predict`), they act as identity layers (no-op).
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
        layers.RandomRotation(0.2, name="aug_rotate"), # Rotates up to +/- 20% of 360 degrees (+/- 72 deg)
        layers.RandomZoom(0.1, name="aug_zoom"), # Zoom in/out up to 10%
        layers.RandomBrightness(0.1, name="aug_brightness"), # Adjust brightness up to 10%
    ], name="data_augmentation")
    
    print("\n--- Step 3: Setting Up Transfer Learning Base ---")
    # We use MobileNetV2 pretrained on ImageNet as our backbone.
    # include_top=False discards the final ImageNet classification layer (top) 
    # so we can append our custom classifier head tailored to our plant classes.
    # pooling='avg' appends a GlobalAveragePooling2D layer to flatten the convolutional features.
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # We freeze the base model parameters so we do not compute gradients or update 
    # their weights during backpropagation. This preserves the ImageNet feature detector representation.
    base_model.trainable = False
    print("MobileNetV2 base model frozen.")
    
    print("\n--- Step 4: Constructing the Custom Model ---")
    # We construct the model architecture using the Keras Functional API.
    # 1. Define input layer matching the expected shape
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    # 2. Apply random data augmentation
    x = data_augmentation(inputs)
    # 3. Preprocess input. MobileNetV2 expects values scaled from [0, 255] to [-1, 1].
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    # 4. Extract features using base model.
    # Setting training=False ensures BatchNorm statistics are NOT updated during training,
    # which is crucial for transfer learning inference consistency.
    x = base_model(x, training=False)
    # 5. Add custom classifier head.
    x = layers.Dense(128, activation='relu', name="dense_head")(x)
    x = layers.Dropout(0.3, name="dropout_head")(x) # Regularization to prevent dense layer overfitting
    outputs = layers.Dense(len(class_names), activation='softmax', name="predictions")(x)
    
    # Assemble model
    model = models.Model(inputs, outputs)
    
    # Compile the model
    # Adam optimizer is used as it adapts learning rates per parameter for stable convergence.
    # categorical_crossentropy loss is standard for multi-class classification with one-hot targets.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    print("\n--- Step 5: Training Model Classifier Head ---")
    # Callbacks are active actions during model training
    callbacks = [
        # EarlyStopping monitors validation loss and stops training early if it doesn't improve
        # for `patience` consecutive epochs, reverting back to the weights that yielded the lowest val loss.
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        # ModelCheckpoint automatically saves the best performing model weights to disk.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Fit the model on training data
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print("\n--- Step 6: Generating Training Curves ---")
    # Plot accuracy and loss curves for train vs validation splits
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    curves_path = os.path.join(EVAL_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved training curves to: {curves_path}")
    
    print("\n--- Step 7: Evaluating Performance on Test Dataset ---")
    # Load the best saved model (this ensures we evaluate the optimal checkpoint)
    best_model = tf.keras.models.load_model(MODEL_PATH)
    
    # Extract actual test classes and generate predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        preds = best_model.predict(images, verbose=0)
        # argmax converts one-hot representation back to class index
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Generate and save the classification report (precision, recall, f1-score)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    report_path = os.path.join(EVAL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to: {report_path}")
    
    # Generate and save the confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Greens', # Green theme for LeafGuard
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    cm_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")
    print("\nModel training and evaluation successfully completed!")

if __name__ == "__main__":
    main()
