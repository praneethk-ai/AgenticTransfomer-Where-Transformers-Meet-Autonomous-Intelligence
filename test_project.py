import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from transformer import Transformer
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_synthetic_data(vocab_size, seq_length, num_samples):
    """Create synthetic data for testing"""
    src_data = torch.randint(1, vocab_size, (num_samples, seq_length))
    tgt_data = torch.randint(1, vocab_size, (num_samples, seq_length))
    return src_data, tgt_data

def train_model(model, train_data, num_epochs=5, batch_size=32, learning_rate=0.001):
    """Train the model and return training history"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    src_data, tgt_data = train_data
    num_batches = len(src_data) // batch_size
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(src_data), batch_size):
            batch_src = src_data[i:i+batch_size]
            batch_tgt = tgt_data[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_src, batch_tgt[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), batch_tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return history

def plot_training_history(history):
    """Plot training loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def test_inference(model, test_data, max_length=50):
    """Test model inference"""
    model.eval()
    src_data, _ = test_data
    
    with torch.no_grad():
        # Generate predictions
        predictions = []
        for src in src_data:
            # Initialize target sequence with start token
            tgt = torch.ones(1, 1, dtype=torch.long) * 1  # Assuming 1 is start token
            
            for _ in range(max_length):
                output = model(src.unsqueeze(0), tgt)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=-1)
                
                if next_token.item() == 2:  # Assuming 2 is end token
                    break
            
            predictions.append(tgt.squeeze().tolist())
    
    return predictions

def benchmark_model(model, vocab_size, input_size=(32, 50), num_runs=100):
    """Benchmark model performance"""
    model.eval()
    src = torch.randint(1, vocab_size, input_size)
    tgt = torch.randint(1, vocab_size, input_size)
    
    # Warmup
    for _ in range(10):
        _ = model(src, tgt)
    
    # Benchmark forward pass
    forward_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model(src, tgt)
        forward_times.append(time.time() - start_time)
    
    # Benchmark backward pass
    backward_times = []
    for _ in range(num_runs):
        # Convert to float for gradient computation
        src_float = src.float()
        tgt_float = tgt.float()
        src_float.requires_grad_(True)
        tgt_float.requires_grad_(True)
        
        start_time = time.time()
        output = model(src, tgt)  # Use original integer tensors for model input
        loss = output.mean()
        loss.backward()
        backward_times.append(time.time() - start_time)
    
    return {
        'forward_mean': np.mean(forward_times),
        'forward_std': np.std(forward_times),
        'backward_mean': np.mean(backward_times),
        'backward_std': np.std(backward_times)
    }

def main():
    # Set random seed
    set_seed(42)
    
    # Model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 1024
    max_seq_length = 100
    
    # Create model
    print("Creating model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length
    )
    
    # Create synthetic data
    print("Creating synthetic data...")
    train_data = create_synthetic_data(src_vocab_size, max_seq_length, 1000)
    test_data = create_synthetic_data(src_vocab_size, max_seq_length, 100)
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, train_data, num_epochs=5)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Test inference
    print("\nTesting inference...")
    predictions = test_inference(model, test_data)
    print(f"Generated {len(predictions)} sequences")
    
    # Benchmark model
    print("\nBenchmarking model...")
    benchmark_results = benchmark_model(model, src_vocab_size)
    print("\nBenchmark Results:")
    print(f"Forward pass: {benchmark_results['forward_mean']*1000:.2f}ms ± {benchmark_results['forward_std']*1000:.2f}ms")
    print(f"Backward pass: {benchmark_results['backward_mean']*1000:.2f}ms ± {benchmark_results['backward_std']*1000:.2f}ms")
    
    # Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_encoder_layers': num_encoder_layers,
        'num_decoder_layers': num_decoder_layers,
        'd_ff': d_ff,
        'max_seq_length': max_seq_length
    }, 'transformer_model.pt')
    print("Model saved as 'transformer_model.pt'")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 