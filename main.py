import torch
from torch.utils.data import DataLoader, TensorDataset
from model import LongformerAPTDetector
from preprocess import StreamPreprocessor
from sklearn.metrics import classification_report

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load and Preprocess
    preprocessor = StreamPreprocessor(window_size=10)
    X, y = preprocessor.process_csv('data/cic_apt_2024.csv')
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Initialize Model (0.84M params)
    model = LongformerAPTDetector(input_dim=X.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 3. Training Loop
    model.train()
    for epoch in range(50):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed.")

    # 4. Save Model
    torch.save(model.state_dict(), 'acfd_transformer_v1.pth')

if __name__ == "__main__":
    main()
