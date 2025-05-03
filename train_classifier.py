import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Filter valid samples (length == 42)
valid_data = []
valid_labels = []

for i in range(len(data_dict['data'])):
    item = data_dict['data'][i]
    if isinstance(item, (list, np.ndarray)) and len(item) == 42:
        valid_data.append(item)
        valid_labels.append(data_dict['labels'][i])

# Convert to NumPy arrays
data = np.array(valid_data, dtype=np.float32)
labels = np.array(valid_labels)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
