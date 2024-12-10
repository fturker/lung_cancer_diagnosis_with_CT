import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def create_label_map(data_dir):
    """
    Verilen klasör için bir etiketleme haritası oluşturur.
    Args:
        data_dir (str): Verilerin bulunduğu ana dizin (örneğin, 'train').
    Returns:
        dict: Dizin adlarını etiketlerle eşleyen bir sözlük.
    """
    # Ana dizindeki klasör adlarını al
    classes = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    classes.sort()

    # Dizin adlarını etiketlerle eşleştir
    label_map = {class_name: idx for idx, class_name in enumerate(classes)}

    return label_map

def label_data(data_dir, label_map):
    """
    Dosya yollarını ve sınıf etiketlerini birleştirir.
    Args:
        data_dir (str): Verilerin bulunduğu ana dizin.
        label_map (dict): Sınıf etiket haritası.
    Returns:
        list: Dosya yolları ve etiketlerden oluşan bir liste.
    """
    labeled_data = []
    for class_name, label in label_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path):
                    labeled_data.append((file_path, label))
    return labeled_data


def resize_with_aspect_ratio(image, target_size=(512, 512), padding_color=(0, 0, 0)):
    """
    Görüntüyü en-boy oranını koruyarak yeniden boyutlandırır ve pad ekler.

    Args:
        image (numpy array): Giriş görüntüsü.
        target_size (tuple): (genişlik, yükseklik) hedef boyut.
        padding_color (tuple): Pad eklemek için kullanılacak renk (varsayılan siyah).

    Returns:
        numpy array: Yeniden boyutlandırılmış görüntü.
    """
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    # Aspect ratio'yu koruyarak yeniden boyutlandır
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Yeni boyutlara göre padding hesapla
    pad_width = (target_width - new_width) // 2
    pad_height = (target_height - new_height) // 2

    # Pad ekle (üst, alt, sol, sağ)
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_height,
        target_height - new_height - pad_height,
        pad_width,
        target_width - new_width - pad_width,
        cv2.BORDER_CONSTANT,
        value=padding_color
    )

    return padded_image


def process_and_save_images(labeled_data, output_dir, target_size=(512, 512)):
    """
    Labeled data üzerinde ön işlemler gerçekleştirir ve işlenmiş görüntüleri kaydeder.
    Args:
        labeled_data (list): Dosya yolları ve etiketlerden oluşan liste.
        output_dir (str): İşlenmiş görüntülerin kaydedileceği klasör.
        target_size (tuple): (genişlik, yükseklik) hedef boyut.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = []
    labels = []
    for file_path, label in labeled_data:
        try:
            # Görüntüyü yükle
            image = cv2.imread(file_path)
            if image is None:
                continue

            # Görüntüyü yeniden boyutlandır
            processed_image = resize_with_aspect_ratio(image, target_size)

            # Görüntüyü grileştirir
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

            # Histogram eşitleme
            processed_image = cv2.equalizeHist(processed_image)

            # Bilateral filtre ile gürültü giderme
            processed_image = cv2.bilateralFilter(processed_image, 5, 25, 25)

            # Görüntüyü normalize et
            processed_image = processed_image.astype('float32') / 255.0

            # Resimleri ve etiketleri listeye ekle
            images.append(processed_image)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Görüntüleri numpy array'e çevir
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Görüntüleri kanal boyutuna sahip hale getirir

    # Etiketleri numpy array'e çevir
    labels = np.array(labels)

    return images, labels


def build_cnn_model(input_shape=(512, 512, 1), num_classes=2):
    """
   b asit bir CNN modelini oluşturur.
    Args:
        input_shape (tuple): Modelin beklediği giriş boyutu.
        num_classes (int): Çıktı sınıf sayısı.
    Returns:
        keras Model: Oluşturulmuş CNN modeli.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(64, (3, 3), activation='relu',),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(128, (3, 3), activation='relu',),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Softmax kullanarak sınıf tahmini yapılır
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Kategorik sınıf kaybı
                  metrics=['accuracy'])

    return model

# Kullanılan veri seti https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
data_directory = "YOUR_DATASET_PATH"  # Ana verinin olduğu dizin
output_directory = "YOUR_OUTPUT_PATH"
label_map = create_label_map(data_directory)
labeled_data = label_data(data_directory, label_map)

# Görüntüleri işle ve kaydet
images, labels = process_and_save_images(labeled_data, output_directory, target_size=(512, 512))

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Modeli oluştur
model = build_cnn_model(input_shape=(512, 512, 1), num_classes=len(label_map))

# Modeli eğit
model.fit(X_train, y_train, epochs=9, batch_size=32, validation_data=(X_test, y_test))

# Etiketleme haritasını ve veriyi incele
print("Label Map:")
print(label_map)
print("\nSample Labeled Data:")
print(labeled_data)
print(len(labeled_data))

# Modeli değerlendir
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Modeli kaydet
model.save('lung_cancer_model.keras')
