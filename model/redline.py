# -*- coding: utf-8 -*-
"""
Improved IndicBERT Training with Class Balance Handling
"""

# ========================
# Additional imports for class balancing
# ========================
!pip
install - q
imbalanced - learn

from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import torch.nn as nn

# ========================
# All your existing code up to the dataset creation remains the same
# I'll show the modifications starting from after dataset creation
# ========================

# ... [Previous code remains exactly the same until dataset creation] ...

# ========================
# ENHANCED: Class Balance Analysis and Handling
# ========================
print( "\n=== CLASS BALANCE ANALYSIS ===" )
class_counts = train_df['label'].value_counts().sort_index()
total_samples = len( train_df )

print( "Class distribution:" )
for label_id, count in class_counts.items() :
    label_name = id2label[label_id]
    percentage = (count / total_samples) * 100
    print( f"  {label_name}: {count} samples ({percentage:.1f}%)" )

# Identify severely underrepresented classes
min_samples_threshold = 100  # Minimum samples for good performance
underrepresented = class_counts[class_counts < min_samples_threshold]

if len( underrepresented ) > 0 :
    print( f"\n⚠️ Underrepresented classes (< {min_samples_threshold} samples):" )
    for label_id, count in underrepresented.items() :
        print( f"  {id2label[label_id]}: {count} samples" )

# ========================
# SOLUTION 1: Compute Class Weights
# ========================
print( "\n=== COMPUTING CLASS WEIGHTS ===" )
y_train = train_df["label"].values
unique_classes = np.unique( y_train )

# Calculate balanced class weights
class_weights = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=y_train
)

class_weight_dict = {i : weight for i, weight in enumerate( class_weights )}
print( "Class weights (to handle imbalance):" )
for label_id, weight in class_weight_dict.items() :
    print( f"  {id2label[label_id]}: {weight:.3f}" )


# ========================
# SOLUTION 2: Data Augmentation for Minority Classes
# ========================
def augment_minority_classes ( df, min_samples=80 ) :
    """Simple data augmentation for minority classes"""
    augmented_data = []

    for label_id in df['label'].unique() :
        class_data = df[df['label'] == label_id].copy()
        current_count = len( class_data )

        if current_count < min_samples :
            needed_samples = min_samples - current_count
            print( f"Augmenting {id2label[label_id]}: {current_count} → {min_samples} samples" )

            # Simple augmentation: repeat samples with slight modifications
            for i in range( needed_samples ) :
                # Randomly select a sample from this class
                sample = class_data.sample( 1 ).copy()
                # Add slight variations (you can enhance this)
                original_text = sample['text'].iloc[0]
                # Simple augmentation: add synonyms or rephrase (basic version)
                augmented_text = original_text  # In practice, you'd apply more sophisticated augmentation
                sample['text'] = augmented_text
                augmented_data.append( sample )

    if augmented_data :
        augmented_df = pd.concat( [df] + augmented_data, ignore_index=True )
        print( f"Dataset augmented: {len( df )} → {len( augmented_df )} samples" )
        return augmented_df
    return df


# Apply augmentation (optional - comment out if you prefer class weights only)
print( "\n=== DATA AUGMENTATION ===" )
# train_df = augment_minority_classes(train_df)  # Uncomment to use

# Update datasets after potential augmentation
train_dataset = Dataset.from_pandas( train_df[['text', 'label']] )
dataset = DatasetDict( {"train" : train_dataset, "test" : test_dataset} )
tokenized_datasets = dataset.map( tokenize_function, batched=True )


# ========================
# ENHANCED: Custom Trainer with Class Weights
# ========================
class WeightedTrainer( Trainer ) :
    def compute_loss ( self, model, inputs, return_outputs=False, **kwargs ) :
        """
        Custom loss function with class weights
        """
        labels = inputs.get( "labels" )
        outputs = model( **inputs )
        logits = outputs.get( "logits" )

        if labels is not None :
            # Apply class weights
            device = logits.device
            weights = torch.tensor( list( class_weight_dict.values() ), dtype=torch.float, device=device )
            loss_fct = nn.CrossEntropyLoss( weight=weights )
            loss = loss_fct( logits.view( -1, self.model.config.num_labels ), labels.view( -1 ) )
        else :
            loss = outputs.loss if hasattr( outputs, 'loss' ) else None

        return (loss, outputs) if return_outputs else loss


# ========================
# ENHANCED: Training Configuration
# ========================
# More aggressive training for imbalanced data
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,  # Slightly higher learning rate
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # More epochs
    weight_decay=0.01,
    warmup_ratio=0.1,  # Add warmup
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # Focus on macro F1 for balanced performance
    greater_is_better=True,
    save_total_limit=3,
    report_to="none",
    seed=42,
    # Additional parameters for better training
    fp16=True,  # Mixed precision for faster training
    dataloader_num_workers=2,
    group_by_length=True,  # Group similar lengths for efficiency
)


# ========================
# ENHANCED: Metrics Function
# ========================
def compute_enhanced_metrics ( pred ) :
    labels = pred.label_ids
    preds = pred.predictions.argmax( -1 )

    # Calculate all metrics with zero_division handling
    precision, recall, f1, _ = precision_recall_fscore_support( labels, preds, average="weighted", zero_division=0 )
    macro_f1 = precision_recall_fscore_support( labels, preds, average="macro", zero_division=0 )[2]
    micro_f1 = precision_recall_fscore_support( labels, preds, average="micro", zero_division=0 )[2]
    acc = accuracy_score( labels, preds )

    # Per-class metrics for detailed analysis
    per_class_f1 = precision_recall_fscore_support( labels, preds, average=None, zero_division=0 )[2]
    per_class_precision = precision_recall_fscore_support( labels, preds, average=None, zero_division=0 )[0]
    per_class_recall = precision_recall_fscore_support( labels, preds, average=None, zero_division=0 )[1]

    metrics = {
        "accuracy" : acc,
        "f1" : f1,
        "f1_macro" : macro_f1,
        "f1_micro" : micro_f1,
        "precision" : precision,
        "recall" : recall
    }

    # Add per-class metrics
    for i, (class_f1, class_prec, class_rec) in enumerate(
            zip( per_class_f1, per_class_precision, per_class_recall ) ) :
        if i < len( id2label ) :
            class_name = id2label[i].replace( ' ', '_' )  # Replace spaces for valid metric names
            metrics[f"f1_{class_name}"] = class_f1
            metrics[f"precision_{class_name}"] = class_prec
            metrics[f"recall_{class_name}"] = class_rec

    return metrics


# ========================
# ENHANCED: Training with Custom Trainer
# ========================
trainer = WeightedTrainer(  # Use custom trainer with class weights
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_enhanced_metrics
)

print( "\n🚀 Starting enhanced training with class balance handling..." )
trainer.train()

# ========================
# ENHANCED: Evaluation with Detailed Analysis
# ========================
print( "\n=== ENHANCED EVALUATION ===" )
results = trainer.evaluate()

# Print all metrics
print( "Overall Metrics:" )
for key, value in results.items() :
    if not key.startswith( ('f1_', 'precision_', 'recall_') ) or key in ['f1_macro', 'f1_micro'] :
        print( f"  {key}: {value:.4f}" )

print( "\nPer-Class Performance:" )
for i, class_name in id2label.items() :
    class_name_clean = class_name.replace( ' ', '_' )
    f1_key = f"eval_f1_{class_name_clean}"
    prec_key = f"eval_precision_{class_name_clean}"
    rec_key = f"eval_recall_{class_name_clean}"

    f1_val = results.get( f1_key, 0 )
    prec_val = results.get( prec_key, 0 )
    rec_val = results.get( rec_key, 0 )

    print( f"  {class_name}:" )
    print( f"    F1: {f1_val:.3f}, Precision: {prec_val:.3f}, Recall: {rec_val:.3f}" )

# ========================
# ENHANCED: Error Analysis
# ========================
print( "\n=== ERROR ANALYSIS ===" )
preds = trainer.predict( tokenized_datasets["test"] )
y_true = preds.label_ids
y_pred = np.argmax( preds.predictions, axis=1 )
pred_probs = torch.nn.functional.softmax( torch.tensor( preds.predictions ), dim=1 )

# Find misclassified samples for each class
print( "Misclassification Analysis:" )
for true_label in range( len( id2label ) ) :
    true_indices = np.where( y_true == true_label )[0]
    if len( true_indices ) == 0 :
        continue

    misclassified = true_indices[y_pred[true_indices] != true_label]
    accuracy_for_class = 1 - (len( misclassified ) / len( true_indices ))

    print( f"\n{id2label[true_label]}:" )
    print( f"  Samples: {len( true_indices )}, Correct: {len( true_indices ) - len( misclassified )}" )
    print( f"  Class Accuracy: {accuracy_for_class:.3f}" )

    if len( misclassified ) > 0 :
        # Show what this class is being confused with
        confused_with = y_pred[misclassified]
        confusion_counts = np.bincount( confused_with, minlength=len( id2label ) )

        print( "  Most confused with:" )
        for confused_label, count in enumerate( confusion_counts ) :
            if count > 0 and confused_label != true_label :
                print( f"    {id2label[confused_label]}: {count} times" )

# ========================
# ENHANCED: Confidence Analysis
# ========================
confidence_scores = torch.max( pred_probs, dim=1 )[0].numpy()

print( f"\n=== CONFIDENCE ANALYSIS ===" )
print( f"Overall confidence statistics:" )
print( f"  Mean: {confidence_scores.mean():.3f}" )
print( f"  Median: {np.median( confidence_scores ):.3f}" )
print( f"  Min: {confidence_scores.min():.3f}" )
print( f"  Max: {confidence_scores.max():.3f}" )

# Confidence by correctness
correct_mask = y_true == y_pred
correct_confidence = confidence_scores[correct_mask]
incorrect_confidence = confidence_scores[~correct_mask]

if len( correct_confidence ) > 0 :
    print( f"Correct predictions confidence: {correct_confidence.mean():.3f}" )
if len( incorrect_confidence ) > 0 :
    print( f"Incorrect predictions confidence: {incorrect_confidence.mean():.3f}" )

# ========================
# VISUALIZATION: Enhanced Plots
# ========================
fig, axes = plt.subplots( 2, 3, figsize=(18, 12) )

# 1. Confusion Matrix
cm = confusion_matrix( y_true, y_pred, labels=list( id2label.keys() ) )
im1 = axes[0, 0].imshow( cm, interpolation='nearest', cmap='Blues' )
axes[0, 0].set_title( 'Confusion Matrix' )
tick_marks = np.arange( len( id2label ) )
axes[0, 0].set_xticks( tick_marks )
axes[0, 0].set_yticks( tick_marks )
axes[0, 0].set_xticklabels( list( id2label.values() ), rotation=45 )
axes[0, 0].set_yticklabels( list( id2label.values() ) )

# Add text annotations
thresh = cm.max() / 2.
for i in range( cm.shape[0] ) :
    for j in range( cm.shape[1] ) :
        axes[0, 0].text( j, i, format( cm[i, j], 'd' ),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black" )

# 2. Per-class F1 scores
class_names = list( id2label.values() )
f1_scores = [results.get( f"eval_f1_{name.replace( ' ', '_' )}", 0 ) for name in class_names]
bars = axes[0, 1].bar( class_names, f1_scores, color=['skyblue' if f1 > 0.3 else 'lightcoral' for f1 in f1_scores] )
axes[0, 1].set_title( 'F1 Score by Class' )
axes[0, 1].set_ylabel( 'F1 Score' )
axes[0, 1].tick_params( axis='x', rotation=45 )
axes[0, 1].set_ylim( 0, 1 )

# Add value labels on bars
for bar, f1 in zip( bars, f1_scores ) :
    height = bar.get_height()
    axes[0, 1].text( bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{f1:.3f}', ha='center', va='bottom' )

# 3. Confidence distribution
axes[0, 2].hist( confidence_scores, bins=50, alpha=0.7, color='lightgreen', edgecolor='black' )
axes[0, 2].axvline( confidence_scores.mean(), color='red', linestyle='--',
                    label=f'Mean: {confidence_scores.mean():.3f}' )
axes[0, 2].set_title( 'Prediction Confidence Distribution' )
axes[0, 2].set_xlabel( 'Confidence Score' )
axes[0, 2].set_ylabel( 'Frequency' )
axes[0, 2].legend()

# 4. Training curves
logs = trainer.state.log_history
train_losses = [x.get( "train_loss" ) for x in logs if "train_loss" in x]
eval_losses = [x.get( "eval_loss" ) for x in logs if "eval_loss" in x]
eval_f1_macro = [x.get( "eval_f1_macro" ) for x in logs if "eval_f1_macro" in x]

if eval_losses :
    epochs = range( 1, len( eval_losses ) + 1 )
    axes[1, 0].plot( epochs, eval_losses, 'b-', marker='o', label='Validation Loss' )
    axes[1, 0].set_title( 'Training Progress - Loss' )
    axes[1, 0].set_xlabel( 'Epoch' )
    axes[1, 0].set_ylabel( 'Loss' )
    axes[1, 0].legend()
    axes[1, 0].grid( True, alpha=0.3 )

# 5. F1 Macro over epochs
if eval_f1_macro :
    axes[1, 1].plot( epochs, eval_f1_macro, 'g-', marker='s', label='Macro F1' )
    axes[1, 1].set_title( 'Training Progress - Macro F1' )
    axes[1, 1].set_xlabel( 'Epoch' )
    axes[1, 1].set_ylabel( 'Macro F1 Score' )
    axes[1, 1].legend()
    axes[1, 1].grid( True, alpha=0.3 )

# 6. Class distribution
class_counts = train_df['label'].value_counts().sort_index()
axes[1, 2].bar( range( len( class_counts ) ), class_counts.values,
                color=['lightblue' if count > 100 else 'orange' for count in class_counts.values] )
axes[1, 2].set_title( 'Training Data Distribution' )
axes[1, 2].set_xlabel( 'Class' )
axes[1, 2].set_ylabel( 'Number of Samples' )
axes[1, 2].set_xticks( range( len( class_counts ) ) )
axes[1, 2].set_xticklabels( [id2label[i] for i in class_counts.index], rotation=45 )

# Add count labels
for i, count in enumerate( class_counts.values ) :
    axes[1, 2].text( i, count + 5, str( count ), ha='center', va='bottom' )

plt.tight_layout()
plt.show()

print( f"\n✅ Enhanced training completed!" )
print( f"📈 Improvement Summary:" )
print( f"  Macro F1: {results['eval_f1_macro']:.4f} (target: >0.5 for balanced performance)" )
print( f"  Weighted F1: {results['eval_f1']:.4f}" )
print( f"  Overall Accuracy: {results['eval_accuracy']:.4f}" )

# Save the enhanced model
trainer.save_model( drive_model_path )
tokenizer.save_pretrained( drive_model_path )