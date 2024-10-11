import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import seaborn as sns

# Streamlit page config
st.set_page_config(page_title="Communication Game Visualization", layout="wide")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
image_size = 64
batch_size = 64
num_epochs = 3
learning_rate = 0.001
vocab_size = 20
message_length = 5
num_distractors = 3
temperature = 1.0

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@st.cache_resource
def load_data():
    train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, index):
        data = self.dataset[index]
        return data[0], data[1], index
    def __len__(self):
        return len(self.dataset)

class CNNEncoder(nn.Module):
    def __init__(self, feature_size=256):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(128 * 8 * 8, feature_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Sender(nn.Module):
    def __init__(self, feature_size=256, vocab_size=20, message_length=5):
        super(Sender, self).__init__()
        self.encoder = CNNEncoder(feature_size=feature_size)
        self.fc = nn.Linear(feature_size, message_length * vocab_size)
        self.vocab_size = vocab_size
        self.message_length = message_length

    def forward(self, images):
        features = self.encoder(images)
        logits = self.fc(features)
        logits = logits.view(-1, self.message_length, self.vocab_size)
        messages = F.gumbel_softmax(logits, tau=temperature, hard=True)
        return messages

class Receiver(nn.Module):
    def __init__(self, vocab_size, message_length):
        super(Receiver, self).__init__()
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.embedding = nn.Linear(vocab_size, 128)
        self.message_processor = nn.GRU(128, 256, batch_first=True)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256)
        )

    def forward(self, messages, candidate_images):
        batch_size = candidate_images.size(0)
        num_candidates = candidate_images.size(1)

        messages = messages.view(batch_size, self.message_length, self.vocab_size)
        embedded = self.embedding(messages)
        _, hidden = self.message_processor(embedded)
        message_features = hidden.squeeze(0)

        candidate_images = candidate_images.view(batch_size * num_candidates, 3, image_size, image_size)
        image_features = self.image_encoder(candidate_images)
        image_features = image_features.view(batch_size, num_candidates, -1)

        message_features = message_features.unsqueeze(1).repeat(1, num_candidates, 1)
        similarities = torch.sum(message_features * image_features, dim=2)
        probs = F.log_softmax(similarities, dim=1)
        return probs

def get_distractors(indices, num_distractors, dataset_length):
    distractor_indices = []
    all_indices = set(range(dataset_length))
    for idx in indices:
        possible_indices = list(all_indices - {idx.item()})
        distractors = random.sample(possible_indices, num_distractors)
        distractor_indices.append(distractors)
    return distractor_indices

def collate_fn_factory(dataset):
    def collate_fn(batch):
        images, labels, indices = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        indices = torch.tensor(indices)
        batch_size = images.size(0)

        distractor_indices = get_distractors(indices, num_distractors, len(dataset))
        candidate_images = []
        for i in range(batch_size):
            candidates = [images[i]]
            for idx in distractor_indices[i]:
                candidate_image = dataset[idx][0]
                candidates.append(candidate_image)
            candidates = torch.stack(candidates)
            candidate_images.append(candidates)
        candidate_images = torch.stack(candidate_images)
        target_positions = torch.zeros(batch_size, dtype=torch.long)
        return images, candidate_images, target_positions, labels
    return collate_fn

@st.cache_resource
def load_models():
    sender = Sender(vocab_size=vocab_size, message_length=message_length).to(device)
    receiver = Receiver(vocab_size=vocab_size, message_length=message_length).to(device)
    return sender, receiver

def train_models(sender, receiver, train_loader, num_epochs):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(list(sender.parameters()) + list(receiver.parameters()), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(num_epochs):
        sender.train()
        receiver.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, candidate_images, target_positions, _ in train_loader:
            images = images.to(device)
            candidate_images = candidate_images.to(device)
            target_positions = target_positions.to(device)

            messages = sender(images)
            probs = receiver(messages, candidate_images)

            loss = criterion(probs, target_positions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(probs, dim=1)
            correct += (preds == target_positions).sum().item()
            total += target_positions.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return train_losses, train_accuracies

def evaluate(model_sender, model_receiver, data_loader):
    model_sender.eval()
    model_receiver.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, candidate_images, target_positions, _ in data_loader:
            images = images.to(device)
            candidate_images = candidate_images.to(device)
            target_positions = target_positions.to(device)

            messages = model_sender(images)
            probs = model_receiver(messages, candidate_images)

            preds = torch.argmax(probs, dim=1)
            correct += (preds == target_positions).sum().item()
            total += target_positions.size(0)
    accuracy = correct / total * 100
    return accuracy

def visualize_training(train_losses, train_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(range(1, num_epochs + 1), train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(range(1, num_epochs + 1), train_accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')

    st.pyplot(fig)

def visualize_tsne(sender, test_loader):
    messages_list = []
    labels_list = []
    sender.eval()
    with torch.no_grad():
        for images, _, _, labels in test_loader:
            images = images.to(device)
            messages = sender(images)
            messages = messages.view(images.size(0), -1)
            messages_list.append(messages.cpu().numpy())
            labels_list.append(labels.numpy())

    messages_array = np.concatenate(messages_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(messages_array)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_array, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    ax.set_title('t-SNE Visualization of Encoded Messages')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    st.pyplot(fig)

def show_image(img, title):
    img = img.cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5  # Denormalize
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    return fig

def get_image_title(index, pred):
    if index == 0:
        return "Target" + (" (Selected!)" if pred == 0 else "")
    elif index == pred:
        return "Selected (Incorrect)"
    else:
        return f"Distractor {index}"

def visualize_message(message):
    message_np = message.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(message_np.reshape(1, -1), ax=ax, cmap='viridis', cbar=True, cbar_kws={'label': 'Symbol Probability'})
    ax.set_title("Encoded Message")
    ax.set_xlabel("Message Dimensions")
    ax.set_yticks([])
    ax.set_ylabel("Symbols")
    return fig

def visualize_communication_game(sender, receiver, test_dataset, device, num_distractors=3):
    sender.eval()
    receiver.eval()
    
    idx = random.randint(0, len(test_dataset) - 1)
    target_image, label, _ = test_dataset[idx]
    target_image = target_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        message = sender(target_image)
    
    distractor_indices = get_distractors(torch.tensor([idx]), num_distractors, len(test_dataset))[0]
    candidate_images = [test_dataset[idx][0]] + [test_dataset[d_idx][0] for d_idx in distractor_indices]
    candidate_images = torch.stack(candidate_images).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = receiver(message, candidate_images)
        pred = torch.argmax(probs, dim=1).item()
    
    st.write("### Communication Game Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(show_image(target_image[0], "Target Image (Sender's View)"))
    with col2:
        st.pyplot(visualize_message(message[0]))
    
    st.write("### Candidate Images (Receiver's View)")
    cols = st.columns(4)
    for i, img in enumerate(candidate_images[0]):
        with cols[i]:
            st.pyplot(show_image(img, get_image_title(i, pred)))
    
    st.write(f"Target Class: {test_dataset.dataset.classes[label]}")
    st.write(f"Receiver's Selection: {'Correct' if pred == 0 else 'Incorrect'}")

def main():
    st.title("Communication Game Visualization")

    train_dataset, test_dataset = load_data()
    train_dataset = IndexedDataset(train_dataset)
    test_dataset = IndexedDataset(test_dataset)

    sender, receiver = load_models()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn_factory(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn_factory(test_dataset))

    if st.button("Train Models"):
        train_losses, train_accuracies = train_models(sender, receiver, train_loader, num_epochs)
        visualize_training(train_losses, train_accuracies)

        test_accuracy = evaluate(sender, receiver, test_loader)
        st.write(f"Test Accuracy: {test_accuracy:.2f}%")

    if st.button("Visualize t-SNE"):
        visualize_tsne(sender, test_loader)

    if st.button("Play Communication Game"):
        visualize_communication_game(sender, receiver, test_dataset, device)

if __name__ == "__main__":
    main()