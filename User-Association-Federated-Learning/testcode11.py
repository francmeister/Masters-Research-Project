import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random

def random_exploration(num_users,num_access_points,num_batches):
    user_association_labels = np.random.randint(2, size=(num_users, num_access_points))
    user_association_labels_ = user_association_labels
    done_sampling = False
    count = 0
    if not np.all(np.sum(user_association_labels,axis=1) <= 1):
         while count < num_batches:
            user_association_labels = np.random.randint(2, size=(num_users, num_access_points))
            if np.all(np.sum(user_association_labels,axis=1) == 1):
                if count == 0:
                   user_association_labels_ = user_association_labels
                else:
                    user_association_labels_ = np.row_stack((user_association_labels_,user_association_labels))
                count+=1
    
    #print('user_association_labels_')
    #print(user_association_labels_)
    user_association_labels_ = user_association_labels_.reshape(num_batches,num_access_points*num_users)
    user_association_labels_ = np.array(user_association_labels_,dtype=np.float32)

    return user_association_labels_
# Generate some random data for demonstration purposes
num_access_points = 3
num_users = 2
num_batches = 1000
num_task_arrival_rate = 1

# Simulate user features (replace this with your actual data)
user_features_large_scale_channel_gains = np.random.rand(num_users*num_batches, num_access_points)
user_features_task_arrival_rates = np.random.rand(num_users*num_batches)



#user_features_large_scale_channel_gains = user_features_large_scale_channel_gains.reshape(1,num_users*num_batches*num_access_points)
# print('user features large scale channel gains after reshape:')
# print(user_features_large_scale_channel_gains)
# print('')


user_features = np.column_stack((user_features_large_scale_channel_gains,user_features_task_arrival_rates))

user_features_reshaped = user_features.reshape(num_batches,(num_access_points+num_task_arrival_rate)*num_users)
user_features_reshaped = np.array(user_features_reshaped,dtype=np.float32)



# Simulate access point features (replace this with your actual data)
access_point_features = torch.rand((num_access_points, 10), dtype=torch.float32)

# Simulate user association labels (binary: 0 or 1)
user_association_labels = random_exploration(num_users,num_access_points,num_batches)


# print('labels')
# print(user_association_labels)
# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(user_features_reshaped, user_association_labels, test_size=0.2, random_state=42)
print('xtrain')
print(X_train)
print('size xtrain: ', len(X_train))
print('ytrain')
print(y_train)
print('size ytrain: ', len(y_train))
print('xtest')
print(X_test)
print('size xtest: ', len(X_test))
print('ytest')
print(y_test)
print('size ytest: ', len(y_test))

# # Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

# # Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Define the neural network model
class UserAssociationModel(nn.Module):
     def __init__(self, input_size, output_size):
         super(UserAssociationModel, self).__init__()
         self.fc1 = nn.Linear(input_size, 64)
         self.relu = nn.ReLU()
         self.fc2 = nn.Linear(64, 32)
         self.fc3 = nn.Linear(32, output_size)
         self.sigmoid = nn.Sigmoid()

     def forward(self, x):
         x = self.relu(self.fc1(x))
         x = self.relu(self.fc2(x))
         x = self.sigmoid(self.fc3(x))
         return x

 # Instantiate the model
model = UserAssociationModel(input_size=len(user_features_reshaped[0]), output_size=num_access_points*num_users)

# # Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training the model
num_epochs = 10
for epoch in range(num_epochs):
    for x_train, y_train in train_loader:
        #inputs = torch.tensor(inputs)
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# # Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    predicted_labels = (test_outputs > 0.5).float()

    accuracy = (predicted_labels == y_test_tensor).float().mean()

    print(f"Test Loss: {test_loss.item()}, Test Accuracy: {accuracy.item() * 100:.2f}%")

# # Make predictions on new data
# new_user_features = torch.rand((5, 10), dtype=torch.float32)  # Replace with your actual new user data
# with torch.no_grad():
#     new_user_predictions = model(new_user_features)
#     new_user_binary_predictions = (new_user_predictions > 0.5).float()

# print("Predictions:")
# print(new_user_binary_predictions.numpy())


    
    