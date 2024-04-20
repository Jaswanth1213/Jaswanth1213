# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Sample data: Equipment usage and maintenance schedules
data = {
    'usage_hours': [100, 200, 300, 400, 500],
    'maintenance_months': [12, 11, 10, 9, 8]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Split data into features (X) and target (y)
X = df[['usage_hours']]
y = df['maintenance_months']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
model = LinearRegression()

# Fit the model on training data
model.fit(X_train, y_train)

# Predict maintenance schedules
y_pred = model.predict(X_test)

# Calculate RMSE (Root Mean Squared Error)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')

# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Usage Hours')
plt.ylabel('Maintenance Months')
plt.title('Equipment Maintenance Prediction')
plt.legend()
plt.show()
