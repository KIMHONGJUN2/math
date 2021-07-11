#데이터 준비
import tensorflow as tf
mnist = tf.keras.datasets.mnist

from tensorflow.keras import datasets,layers,models
(train_images,train_labels), (test_images,test_labels) = mnist.load_data()

train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))
train_images, test_images = train_images / 255.0, test_images/ 255.0

#cnn 구성
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation=('relu')))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

#결과층 모형 구성 (최종 출력값에서 출력 64개를 입력 값으로 받아 0~9 레이블 10개 구성
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

#에포크 지정 후 학습
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5)

#평가
test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
print(test_acc)