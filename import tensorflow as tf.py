import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import json
import os

# 確保 model 資料夾存在
os.makedirs('model', exist_ok=True)

# 載入 Fashion MNIST 資料集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 正規化資料（將像素值縮放到 [0, 1]）
x_train = x_train / 255.0
x_test = x_test / 255.0

# 定義模型：增加層數和單元數以提高準確率
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28), name='flatten_1'),
    layers.Dense(512, activation='relu', name='dense_1'),
    layers.Dense(256, activation='relu', name='dense_2'),
    layers.Dense(128, activation='relu', name='dense_3'),
    layers.Dense(10, activation='softmax', name='dense_4')
])

# 編譯模型：使用 Adam 優化器和較低學習率
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型：增加 epoch 以提高準確率
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# 保存模型為 .h5 格式（用於檢查）
model.save('model/fashion_mnist.h5')

# 保存模型結構為 .json（與 nn_predict.py 相容）
model_json = model.to_json()
model_config = json.loads(model_json)['config']['layers']
arch = [
    {
        'name': layer['config']['name'],
        'type': layer['class_name'],
        'config': {'activation': layer['config'].get('activation', '')},
        'weights': [] if layer['class_name'] == 'Flatten' else [f"{layer['config']['name']}_w", f"{layer['config']['name']}_b"]
    } for layer in model_config
]
with open('model/fashion_mnist.json', 'w') as f:
    json.dump(arch, f, indent=2)

# 保存權重為 .npz
weights = model.get_weights()
np.savez('model/fashion_mnist.npz',
         dense_1_w=weights[0], dense_1_b=weights[1],
         dense_2_w=weights[2], dense_2_b=weights[3],
         dense_3_w=weights[4], dense_3_b=weights[5],
         dense_4_w=weights[6], dense_4_b=weights[7])