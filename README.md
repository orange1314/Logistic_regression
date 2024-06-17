# Logistic Regression Basic Introduction

Logistic Regression is a statistical model used for classification problems, especially suitable for binary classification issues (i.e., target variable has only two possible values, such as 0 and 1, or true and false). Despite its name including the word "regression," logistic regression is a classification method.

## Key Features and Principles of Logistic Regression:

### 1. Basic Concept
The logistic regression model learns the relationship between input features and the target variable to predict the probability of the target variable. Assume we have a binary classification problem where the target variable \( Y \) can take values 0 or 1. We want to predict the value of \( Y \) based on input features \( X = (X_1, X_2, \ldots, X_n) \).

### 2. Transformation Function
Logistic regression uses the logistic function (also called the sigmoid function) to convert the output of linear regression into probability values. The mathematical expression of the logistic function is:
\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]
where \( z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n \) is a linear combination, and \( \beta_i \) are the parameters of the model.

### 3. Predicting Probabilities
Using the logistic function, we can convert the linear combination into the probability that the target variable is 1:
\[ P(Y = 1|X) = \sigma(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n) \]
For binary classification problems, this probability helps us decide the final classification result. For example, if the probability is greater than 0.5, we predict \( Y = 1 \); otherwise, we predict \( Y = 0 \).

### 4. Model Training
The parameters \( \beta_i \) of the logistic regression model are estimated using Maximum Likelihood Estimation (MLE). This involves maximizing the probability of observing the actual values of the target variable, thereby finding the optimal set of parameters.

### 5. Performance Evaluation
The performance of a logistic regression model is usually evaluated using metrics such as confusion matrix, accuracy, recall, and F1 score. These metrics help us understand the model's performance in classification tasks.

### 6. Extension to Multiclass Classification
Logistic regression can also be extended to handle multiclass classification problems, known as multinomial logistic regression or softmax regression. In this case, we use the softmax function to convert the linear combinations into probabilities for multiple classes.

In summary, logistic regression is a simple yet effective classification algorithm widely used in various fields such as medical diagnosis, credit scoring, and marketing. Its advantages include easy interpretability, computational efficiency, and good performance on linearly separable problems.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Generate data points ranging from -10 to 10
x = torch.linspace(-10, 10, 100)
y = sigmoid(x)

# Plot the sigmoid curve
plt.plot(x.numpy(), y.numpy(), label="Sigmoid Function")
plt.title("Sigmoid Function in Logistic Regression")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.legend()
plt.grid()
plt.show()
```

## Statistical Significance of Logistic Regression

### 1. Probability Modeling
The logistic regression model maps input variables to a probability value, making it very suitable for binary classification problems. By predicting the probability of an event occurring, logistic regression provides a quantitative basis for decision-making. For example, in medicine, it can predict the probability of a patient having a certain disease, helping doctors make treatment decisions.

### 2. Interpretability
The parameters (β values) in the logistic regression model have clear interpretative significance. Each parameter represents the impact of the corresponding input variable on the prediction result. Specifically, the larger the coefficient of a variable, the more significant its impact on the target variable. This makes logistic regression advantageous in variable selection and interpretation.

### 3. Hypothesis Testing
Logistic regression provides a framework for hypothesis testing to examine the relationships between variables. For example, likelihood ratio test, Wald test, and score test can be used to test the significance of each parameter. These tests help researchers determine which variables have significant impacts on the model.

### 4. Analysis of Ordinal and Categorical Data
Logistic regression models can be extended to handle ordinal and categorical data. For example, ordinal logistic regression and multinomial logistic regression are used to handle ordinal and multi-category variables, respectively. These models are widely applied in market research and social science research.

### 5. Wide Applications
Logistic regression is applied in multiple fields such as medical diagnosis, financial risk assessment, social science research, and marketing. In these fields, logistic regression models help analysts and decision-makers understand patterns in the data and make informed decisions.

### 6. Multivariate Analysis
Logistic regression can handle multiple independent variables, allowing it to analyze the interactions between variables. This ability is especially important in exploring the relationships between variables in complex systems, such as analyzing the impacts of genes and environmental factors on diseases in biomedical research.

### 7. Handling Imbalanced Data
Logistic regression can handle imbalanced datasets by adjusting the decision threshold and introducing weights. This is crucial in many practical applications, such as fraud detection and rare event prediction.

In summary, logistic regression is of significant importance in statistics, providing an effective tool for handling classification problems with good interpretability and flexibility, and is widely applied in various fields.

## Why Logistic Regression Uses MLE Instead of MSE

### Consideration of Probability Distribution
Logistic regression assumes the target variable follows a Bernoulli distribution, which aligns with the principles of Maximum Likelihood Estimation (MLE). Mean Squared Error (MSE) is suitable for Gaussian-distributed data, whereas the binary variable in logistic regression clearly does not fit this assumption. Therefore, using MLE can more accurately capture the distribution characteristics of binary variables.

### Maximum Likelihood Estimation (MLE)
Maximum Likelihood Estimation (MLE) is a statistical method used to estimate the parameters of a model, making the observed data most likely given the parameters. In other words, MLE attempts to find the parameter values that make the observed data most probable.

### Mathematical Definition of MLE
Suppose we have a set of observations \( \{x_1, x_2, \ldots, x_n\} \) generated from a probability distribution with parameter \( \theta \). The likelihood function \( L(\theta) \) is defined as the joint probability density function of the observed data given the parameter \( \theta \):

\[ L(\theta) = P(x_1, x_2, \ldots, x_n | \theta) = \prod_{i=1}^{n} P(x_i | \theta) \]

where \( P(x_i | \theta) \) is the probability of data \( x_i \) given the parameter \( \theta \).

MLE tries to find the estimate of the parameter \( \hat{\theta} \) that maximizes the likelihood function \( L(\theta) \):

\[ \hat{\theta} = \arg \max_{\theta} L(\theta) \]

Since the logarithm is a monotonically increasing function, we usually maximize the log-likelihood function to simplify calculations:

\[ \ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log P(x_i | \theta) \]

Therefore, the estimate of \( \theta \) using MLE can also be expressed as:

\[ \hat{\theta} = \arg \max_{\theta} \ell(\theta) \]

### Application in Logistic Regression
In logistic regression, we assume the output variable \( y \) follows a Bernoulli distribution, with the probability determined by the linear combination of input features \( X \) through the sigmoid function:

\[ P(y = 1 | X, \theta) = \sigma(X \cdot \theta) = \frac{1}{1 + e^{-(X \cdot \theta)}} \]

where \( \theta \) are the parameters to be estimated, including weights and biases.

For a given training dataset \( \{(X_i, y_i)\} \), the likelihood function of the logistic regression model is:

\[ L(\theta) = \prod_{i=1}^{m} P(y_i | X_i, \theta) = \prod_{i=1}^{m} \sigma(X_i \cdot \theta)^{y_i} (1 - \sigma(X_i \cdot \theta))^{1 - y_i} \]

The log-likelihood function is:

\[ \ell(\theta) = \sum_{i=1}^{m} \left[ y_i \log(\sigma(X_i \cdot \theta)) + (1 - y_i) \log(1 - \sigma(X_i \cdot \theta)) \right] \]

To maximize the log-likelihood function, we can use gradient descent to iteratively update the parameters \( \theta \).

In gradient descent, the parameter update formula is:

\[ \theta \leftarrow \theta + \alpha \frac{\partial \ell(\theta)}{\partial \theta} \]

where \( \alpha \) is the learning rate and \( \frac{\partial \ell(\theta)}{\partial \theta} \) is the gradient of the log-likelihood function with respect to the parameter \( \theta \).

## Implementing Logistic Regression Using MLE

### Generating a Simple Binary Classification Dataset

```python


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate binary classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Plot data points (choose the first feature)
plt.scatter(X_train[:, 0], y_train, color='blue', label='Train data')

# Add title and labels
plt.title('Logistic Regression')
plt.xlabel('Feature 1')
plt.ylabel('Class')
plt.legend()
plt.grid()
plt.show()

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, alpha = 0.5)
# Add title and labels
plt.title('data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
```

## Define the Sigmoid Function

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### Step 5: Define the Logistic Regression Model and Likelihood Function

The probability mass function of a Bernoulli distribution, representing the probability that a random variable \( X \) takes value \( x \), is given by:

\[ f_{X}(x) = p^{x}(1-p)^{1-x} = \left\{ \begin{matrix} p & \text{if } x = 1, \\ 1-p & \text{if } x = 0. \end{matrix} \right. \]

In logistic regression, given model parameters \( \mathbf{W} \) and \( \mathbf{b} \), the predicted probability for sample \( i \) is:

\[ \hat{y}_i = \sigma(\hat{\mathbf{W}} \cdot \hat{\mathbf{X}_i} + \hat{b}) = \frac{1}{1 + e^{-(\hat{\mathbf{W}} \cdot \hat{\mathbf{X}_i} + \hat{b})}} \]

The negative log-likelihood loss function can be expressed as:

\[ L(\mathbf{W}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] \]

where:
- \( m \) is the number of samples
- \( y_i \) is the true label
- \( \hat{y}_i \) is the predicted probability

The parameter update formulas for the weights \( \mathbf{W} \) and bias \( b \) using gradient descent are:

\[ \mathbf{W} \leftarrow \mathbf{W} - \alpha \frac{\partial L}{\partial \mathbf{W}} \]
\[ b \leftarrow b - \alpha \frac{\partial L}{\partial b} \]

where:
- \( \alpha \) is the learning rate
- \( \frac{\partial L}{\partial \mathbf{W}} \) and \( \frac{\partial L}{\partial b} \) are the gradients of the loss function with respect to the weights and bias.

These gradients are computed as:

\[ \frac{\partial L}{\partial \mathbf{W}} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i) \mathbf{X}_i \]
\[ \frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i) \]

The implementation of the logistic regression model using MLE is as follows:

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.y = y

        self.losses = []
        for i in range(self.num_iterations):
            loss = self.update_weights()
            self.losses.append(loss)
            if (i+1) % 100 == 0:
                print(f'Epoch [{i+1}/{self.num_iterations}], Loss: {loss:.4f}')

    def update_weights(self):
        # Predict
        linear_model = np.dot(self.X, self.W) + self.b
        y_pred = sigmoid(linear_model)

        # Compute loss (negative log-likelihood loss)
        loss = - (1 / self.m) * np.sum(self.y * np.log(y_pred) + (1 - self.y) * np.log(1 - y_pred))

        # Compute gradients
        dw = (1 / self.m) * np.dot(self.X.T, (y_pred - self.y))
        db = (1 / self.m) * np.sum(y_pred - self.y)

        # Update weights
        self.W -= self.learning_rate * dw
        self.b -= self.learning_rate * db

        return loss

    def predict(self, X):
        linear_model = np.dot(X, self.W) + self.b
        y_pred = sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)
```

```python
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

```python
# Plot data points
plt.scatter(X_train[:, 0], y_train, color='blue', label='Train data')
plt.scatter(X_test[:, 0], y_test, color='green', label='Test data')

# Plot logistic regression curve
x_values = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
y_values = sigmoid(model.W[0] * x_values + model.b)
plt.plot(x_values, y_values, color='red', label='Logistic Regression')

# Add title and labels
plt.title('Logistic Regression')
plt.xlabel('Feature 1')
plt.ylabel('Probability')
plt.legend()
plt.grid()
plt.show()

# Plot loss function curve
plt.plot(range(model.num_iterations), model.losses, color='purple')
plt.title('Loss Function')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid()
plt.show()
```

## Torch Version

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
```

```python
# Generate binary classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(X_train.to(device),y_train.to(device))
test_dataset = TensorDataset(X_test.to(device),y_test.to(device))

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)
```

```python
dim = X_train.shape[1]
```

```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, dim):
        super(LogisticRegressionModel, self).__init__()
        self.h1 = nn.Linear(dim, 1)
    def forward(self, x):
        x = torch.sigmoid(self.h1(x))
        return x
```

```python
from torchinfo import summary
model = LogisticRegressionModel(dim).to(device)
summary(model, input_size=(1, 2))
```

```python
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop

epochs = 100
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)  # Ensure data is on the correct device
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1)

 % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
```

```python
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        y_pred = model(xb)
        predicted = (y_pred > 0.5).float()
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
```

```python
# Visualize results
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(grid).reshape(xx.shape).cpu().numpy()
    plt.contourf(xx, yy, probs, alpha=0.8, levels=[0, 0.5, 1])
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.show()

plot_decision_boundary(model, X_test.cpu().numpy(), y_test.cpu().numpy())
```

This code demonstrates the implementation of logistic regression using MLE in both a manual approach and using the PyTorch framework. The logistic regression model is trained to classify a binary dataset, and the results are visualized along with the decision boundary and loss function curve.
