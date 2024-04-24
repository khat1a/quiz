import ipaddress
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("Darknet.csv")
df = df.drop(["Flow ID", "Timestamp", "Label2"], axis=1)
df = df.dropna()

df['Src IP'] = df['Src IP'].apply(lambda x: int(ipaddress.ip_address(x)))
df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ipaddress.ip_address(x)))

for col in ['Src Port', 'Dst Port', 'Protocol']:
    df[col] = df[col].astype(int)

label_encoder = LabelEncoder()
df['Label1'] = label_encoder.fit_transform(df['Label1'])

df.to_csv("processed.csv", index=False)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

scaler = RobustScaler()
scaled_features = scaler.fit_transform(df.drop('Label1', axis=1))

scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])
scaled_df['Label1'] = df['Label1']

scaled_df.to_csv("scaled.csv", index=False)

scaled_df = pd.read_csv("scaled.csv")

features = scaled_df.drop(['Label1'], axis=1)
label = scaled_df['Label1']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=1000, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
