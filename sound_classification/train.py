# train.py
def train_model(data_dir: str, num_epochs=100):
    """Train the sound detection model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Data augmentation for better generalization
    train_transform = torch.nn.Sequential(
        torchaudio.transforms.TimeMasking(time_mask_param=30),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    )
    
    # Initialize model
    model = SoundDetectorCNN(num_classes=len(classes))
    model.to(device)
    
    # Use mixed precision training for RTX 5080
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = criterion(predictions, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        # Validation
        accuracy = validate_model(model, val_loader, device)
        
        print(f"Epoch {epoch}: Loss={running_loss/len(train_loader):.4f}, "
              f"Accuracy={accuracy:.2%}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'num_classes': len(classes)
            }, 'best_model.pth')
        
        scheduler.step(accuracy)
    
    return model


if __name__ == "__main__":
    train_model('data_dir')