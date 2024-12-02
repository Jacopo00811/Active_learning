import torchvision
import torchvision.transforms as transforms
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import PIL.Image

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_set = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# Extract images and labels
images = []
labels = []
for img, label in train_set:
    images.append(img.numpy())
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Flatten images
flattened_images = images.reshape(len(images), -1)

# Perform PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(flattened_images)

# Create Plotly 3D scatter plot with images
def create_image_hover(img):
    # Denormalize and convert to PIL Image
    img = img * 0.5 + 0.5  # Reverse normalization
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    pil_img = PIL.Image.fromarray(img)
    
    # Convert to base64
    import io
    import base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" width="100">'

# Create traces for different classes
traces = []
unique_labels = np.unique(labels)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

for label in unique_labels:
    mask = labels == label
    trace = go.Scatter3d(
        x=pca_result[mask, 0], 
        y=pca_result[mask, 1], 
        z=pca_result[mask, 2],
        mode='markers',
        name=class_names[label],
        text=[create_image_hover(img) for img in images[mask]],
        hoverinfo='text',
        marker=dict(
            size=3,
            opacity=0.7
        )
    )
    traces.append(trace)

# Create layout
layout = go.Layout(
    title='CIFAR-10 Images in PCA Space',
    scene=dict(
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component', 
        zaxis_title='Third Principal Component'
    )
)

# Create figure and plot
fig = go.Figure(data=traces, layout=layout)
fig.write_html("cifar10_pca_visualization.html")
print("Visualization saved as cifar10_pca_visualization.html")

# Print variance explained
print("\nVariance Explained by Components:")
print(f"1st Component: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"2nd Component: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"3rd Component: {pca.explained_variance_ratio_[2]*100:.2f}%")
