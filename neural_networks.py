import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        
        # Define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.hidden_activations = None
        self.gradients = None
        
    def activate(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activate_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # Forward pass, apply layers to input X
        # Store activations for visualization
        self.Z1 = X @ self.W1 + self.b1
        self.hidden_activations = self.activate(self.Z1)
        
        self.Z2 = self.hidden_activations @ self.W2 + self.b2
        out = self.activate(self.Z2)
        
        return out

    def backward(self, X, y):
        # Compute gradients using chain rule
        out = self.activate(self.Z2)
        dZ2 = (out - y) * self.activate_derivative(self.Z2)
        dW2 = self.hidden_activations.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = (dZ2 @ self.W2.T) * self.activate_derivative(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        # Store gradients for visualization
        self.gradients = np.abs(dW1)

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.hidden_activations
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Layer Features")

    # Hyperplane visualization in the hidden space
    ax_hidden.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_hidden.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Distorted input space transformed by the hidden layer
    distorted = np.tanh(hidden_features)
    ax_hidden.scatter(distorted[:, 0], distorted[:, 1], c=y.ravel(), alpha=0.5)

    # Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid)
    preds = preds.reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=50, cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Space Decision Boundary")
    
    # Visualize features and gradients as circles and edges 
    gradients = mlp.gradients
    for i in range(gradients.shape[0]):
        circle = Circle((X[i, 0], X[i, 1]), radius=0.05, edgecolor='k', facecolor='none', linewidth=np.linalg.norm(gradients[i]))
        ax_gradient.add_patch(circle)
    ax_gradient.set_xlim(x_min, x_max)
    ax_gradient.set_ylim(y_min, y_max)
    ax_gradient.set_title("Gradients")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)