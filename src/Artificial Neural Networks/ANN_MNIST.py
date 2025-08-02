import tensorflow as tf
from docutils.nodes import legend
from tensorflow.python.ops.numpy_ops.np_array_ops import newaxis

print(tf.__version__)

import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(f"X_train.shape: {x_train.shape}")

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

plt.plot(r.history['loss'], label='loss');
plt.plot(r.history['val_loss'], label='val_loss');
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy');
plt.plot(r.history['val_accuracy'], label='val_accuracy');
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()

p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

# Show some misclassified examples
def show_random_misclassified(x_test, y_test, p_test):
    misclassified_idx = np.where(p_test != y_test)[0]
    if len(misclassified_idx) == 0:
        print("No misclassified samples")
        return

    i = np.random.choice(misclassified_idx)
    plt.imshow(x_test[i], cmap='gray')
    plt.title("True label: %s Predicted label: %s" %(y_test[i], p_test[i]))
    plt.show()

while input("Show another misclassified image? (y/n): ").lower() == 'y':
    show_random_misclassified(x_test, y_test, p_test)