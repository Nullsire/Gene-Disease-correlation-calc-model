import re
import matplotlib.pyplot as plt
import os

def parse_log_file(file_path):
    epochs = []
    train_rmses = []
    val_rmses = []
    
    genes = []
    scores = []
    
    reading_predictions = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        
        # Parse Training Progress
        # Example: Epoch [1/25], Train RMSE: 0.1385, Val RMSE: 0.0917
        epoch_match = re.search(r'Epoch \[(\d+)/\d+\], Train RMSE: ([\d\.]+), Val RMSE: ([\d\.]+)', line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))
            train_rmses.append(float(epoch_match.group(2)))
            val_rmses.append(float(epoch_match.group(3)))
            continue
            
        # Parse Prediction Results
        if "Predicted Top Genes" in line or "Predicted_Score" in line:
            reading_predictions = True
            continue
            
        if reading_predictions:
            # Example: 839   ENSG00000091831         0.824326
            # We expect Gene_ID and Predicted_Score. 
            # The line might look like index, gene_id, score or just gene_id, score depending on how pandas printed it.
            # Based on log.txt: "839   ENSG00000091831         0.824326"
            # It seems to be Index GeneID Score
            parts = line.split()
            if len(parts) >= 2:
                # Check if the last part is a float (score) and second to last is gene id
                try:
                    score = float(parts[-1])
                    gene_id = parts[-2]
                    # Simple check to look like a gene id
                    if gene_id.startswith("ENSG"): 
                        genes.append(gene_id)
                        scores.append(score)
                except ValueError:
                    pass

    return epochs, train_rmses, val_rmses, genes, scores

def plot_training_process(epochs, train_rmses, val_rmses, output_path='training_process.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmses, label='Train RMSE', marker='o')
    plt.plot(epochs, val_rmses, label='Val RMSE', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training Process: RMSE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Training process plot saved to {output_path}")
    plt.close()

def plot_prediction_results(genes, scores, output_path='prediction_results.png'):
    if not genes:
        print("No prediction data found to plot.")
        return

    plt.figure(figsize=(12, 8))
    # Horizontal bar chart for better readability of gene IDs
    y_pos = range(len(genes))
    
    # Sort by score just in case, though log seems sorted
    # data = sorted(zip(genes, scores), key=lambda x: x[1], reverse=False) # Ascending for barh
    # genes_sorted, scores_sorted = zip(*data)
    
    # Actually log is top to bottom (high to low). For barh, we usually want high at top, 
    # but pyplot plots 0 at bottom. So we reverse the lists to have highest score at the top of the chart.
    genes_reversed = genes[::-1]
    scores_reversed = scores[::-1]
    
    plt.barh(range(len(genes_reversed)), scores_reversed, align='center')
    plt.yticks(range(len(genes_reversed)), genes_reversed)
    plt.xlabel('Predicted Score')
    plt.title('Top Predicted Genes for EFO_0000305')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Prediction results plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    log_file = "log.txt"
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
    else:
        epochs, train_rmses, val_rmses, genes, scores = parse_log_file(log_file)
        
        if epochs:
            plot_training_process(epochs, train_rmses, val_rmses)
        else:
            print("No training data found in log.")
            
        if genes:
            plot_prediction_results(genes, scores)
        else:
            print("No prediction data found in log.")
