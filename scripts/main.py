import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def load_data(file_path):
    """Load data from a text file where each line has 441 cell energies, true energy, and class."""
    data = []
    labels_energy = []
    labels_class = []
    
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()  # Split on any whitespace
            # print(f"Line has {len(values)} values")  # Debug: Show value count per line
            if len(values) != 1683:
                # print("Skipping line")  # Debug: Indicate skipped lines
                continue
            try:
                energies = list(map(float, values[:1681]))  # 441 cell energies
                energy_label = float(values[1681])          # True energy
                class_label = int(float(values[1682]))      # Class label
                data.append(energies)
                labels_energy.append(energy_label)
                labels_class.append(class_label)
            except ValueError as e:
                print(f"Error parsing line: {e}")  # Debug: Catch parsing errors
    print(f"Loaded {len(data)} samples")  # Debug: Total samples loaded
    return np.array(data), np.array(labels_energy), np.array(labels_class)

def reshape_data(data):
    """Reshape flat 441 cell energies into a 21x21 grid for each event."""
    d=[]
    for i in range(len(data)):
        d.append(data[i].reshape(-1, 41, 41, 1))
    return data.reshape(-1, 41, 41, 1) # Add channel dimension for CNN compatibility

# Specify your file path (update this to match your actual file)
file_path = '../dataset/calorimeter_event_data.txt'

# Load the data
X, y_energy, y_class = load_data(file_path)

X = reshape_data(X)

# Check if data was loaded
if X.shape[0] > 0:
    # Split into training and testing sets
    X_train, X_test, y_energy_train, y_energy_test, y_class_train, y_class_test = train_test_split(
        X, y_energy, y_class, test_size=0.2, random_state=42
    )
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
else:
    print("No data loaded. Please check the data file and its format.")

# Создание модели с двумя выходами
inputs = tf.keras.Input(shape=(41, 41, 1))
x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Выход для классификации
class_output = tf.keras.layers.Dense(2, activation='softmax', name='class_output')(x)
# Выход для регрессии
energy_output = tf.keras.layers.Dense(1, name='energy_output')(x)

# Определение модели
model = tf.keras.Model(inputs=inputs, outputs=[class_output, energy_output])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss={
        'class_output': 'sparse_categorical_crossentropy',
        'energy_output': 'mean_squared_error'
    },
    metrics={
        'class_output': ['accuracy'],
        'energy_output': ['mae']
    }
)

# Обучение модели
model.fit(
    X_train,
    {'class_output': y_class_train, 'energy_output': y_energy_train},
    epochs=30,
    validation_split=0.2
)

# Оценка на тестовых данных
results = model.evaluate(
    X_test,
    {'class_output': y_class_test, 'energy_output': y_energy_test}
)
print(f"Test results - Classification accuracy: {results[3]}, Energy MAE: {results[4]}")

predictions = model.predict(X_test)

# Extract the two outputs
class_predictions = predictions[0]  # Probabilities for classes, shape: (n_samples, 2)
energy_predictions = predictions[1]  # Predicted energies, shape: (n_samples, 1) or (n_samples,)

# Get predicted class labels (0 for EM, 1 for hadronic)
predicted_classes = np.argmax(class_predictions, axis=1)

residuals = energy_predictions - y_energy_test

positive_class_probs = class_predictions[:, 1]

em_avg = np.mean(X_test[y_class_test == 0], axis=0).squeeze()  # Average grid for EM events
hadronic_avg = np.mean(X_test[y_class_test == 1], axis=0).squeeze()  # Average grid for hadronic events

cm = confusion_matrix(y_class_test, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['EM', 'Hadronic'], yticklabels=['EM', 'Hadronic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

plt.scatter(y_energy_test, energy_predictions, alpha=0.5)
plt.plot([0, 6], [0, 6], 'r--')  # Diagonal line (adjust range as needed)
plt.xlabel('True Energy (GeV)')
plt.ylabel('Predicted Energy (GeV)')
plt.title('Predicted vs. True Energy')

plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel('Residual (Predicted - True Energy)')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')

fpr, tpr, _ = roc_curve(y_class_test, positive_class_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(em_avg, ax=axes[0], cmap='viridis')
axes[0].set_title('Average EM Event')
sns.heatmap(hadronic_avg, ax=axes[1], cmap='viridis')
axes[1].set_title('Average Hadronic Event')
plt.show()
