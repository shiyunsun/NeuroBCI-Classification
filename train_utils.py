import numpy as np
import torch

def randomly_crop(X, crop_len):
    # Randomly crop part of the sequence
    seq_len = X.shape[2]
    start_i = torch.randint(0, seq_len-crop_len+1, (1,))
    crop_idx = torch.arange(crop_len) + start_i
    croppedX = X[:, :, crop_idx]
    return croppedX

def model_train(num_epochs, model, optimizer, dataloader, 
                X_val, y_val, X_test, y_test, seq_len, 
                save=False, mute=False):
    val_acc = []
    val_loss = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for i in range(num_epochs):
        batch_loss = []
        batch_acc = []
        model.train()
        for X_train, y_train in dataloader:
            optimizer.zero_grad()
            
            # Crop training sequence and add noide
            crop_len = np.random.randint(seq_len-10, seq_len+11)
            X_train = X_train[:, :, :seq_len+10]
            X_train = X_train + torch.randn(X_train.shape)/2
            X_train = randomly_crop(X_train, crop_len)
            y_predict = model(X_train)
            loss = crit(y_predict, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            # Log training history for this epoch
            batch_loss.append(loss.item())
            y_predict = torch.argmax(model(X_train), dim=1)
            batch_acc.append((y_predict==y_train).sum().item()/y_train.shape[0])
        
        train_loss.append(np.mean(batch_loss))
        train_acc.append(np.mean(batch_acc))
        
        # Evaluation on the eval set and log info
        model.eval()
        scores = model(X_val[:, :, :seq_len])
        y_predict = torch.argmax(scores, dim=1)
        validation_accuracy = ((y_predict==y_val).sum().item()/y_val.shape[0])
        val_acc.append(validation_accuracy)
        loss = crit(scores, y_val)
        val_loss.append(loss.item())
        
        
        # Evaluation on the test set and log info
        scores = model(X_test[:, :, :seq_len])
        y_predict = torch.argmax(scores, dim=1)
        test_accuracy = ((y_predict==y_test).sum().item()/y_test.shape[0])
        test_acc.append(test_accuracy)
        loss = crit(scores, y_test)
        test_loss.append(loss.item())
        
        # Save the model with the best performance
        if save:
            if validation_accuracy > max(val_acc):
                torch.save(model, "best_val.pth")
            if loss < min(val_loss):
                torch.save(model, "best_loss.pth")

        
        if i%30 == 0 and not mute:
            print("Iter", i)
            print("Training Loss: ", np.mean(train_loss[-30:]))
            print("Training accuracy: ", np.mean(train_acc[-30:]))
            print("Validation Loss: ", np.mean(val_loss[-30:]))
            print("Validation Accuracy: ", np.mean(val_acc[-30:]))
            print("Test Loss: ", np.mean(test_loss[-30:]))
            print("Test Accuracy: ", np.mean(test_acc[-30:]))
            print()
    
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

def plot_history(train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    # Plot the training history
    print(max(test_acc))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'pink', label='Train')
    plt.plot(val_loss, 'mediumspringgreen', label='Validation')
    plt.plot(test_loss, 'skyblue', label='Test')
    plt.title('Loss Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, 'pink', label='Train')
    plt.plot(val_acc, 'mediumspringgreen', label='Validation')
    plt.plot(test_acc, 'skyblue', label='Test')
    plt.title('Accuracy Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return