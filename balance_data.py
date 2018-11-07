import pandas
import time
import numpy
import sklearn


def import_csv(path):
    """import a csv file and shuffle it
    Arguments:
      path {string} -- path to the csv file
    Returns:
      data -- numpy array
    """
    print('importing', path)
    start_time = time.time()

    data_frame = pandas.read_csv(path)
    data_frame = sklearn.utils.shuffle(data_frame)
    data = data_frame.as_matrix()
    print("\nimportation done in:", time.time() - start_time)
    return data

def balance_dataset(dataset, percentage):    
    # first we count and rank the number of samples for each class
    class_happy = 0
    class_sad = 0
    class_angry = 0
    class_disgust = 0
    class_fear = 0
    class_neutral = 0
    class_surprise = 0
    for position in range (dataset.shape[0]):
        if(dataset[position, 0] == 0):
            class_angry += 1
        if(dataset[position, 0] == 1):
            class_disgust += 1
        if(dataset[position, 0] == 2):
            class_fear += 1
        if(dataset[position, 0] == 3):
            class_happy += 1
        if(dataset[position, 0] == 4):
            class_sad += 1
        if(dataset[position, 0] == 5):
            class_surprise += 1
        if(dataset[position, 0] == 6):
            class_neutral += 1

    rank = [class_angry, class_disgust, class_fear, class_happy, class_sad, class_surprise, class_neutral]
    rank.sort()
    print("happy:.....", class_happy)
    print("sad:.......", class_sad)
    print("angry:.....", class_angry)
    print("disgust:...", class_disgust)
    print("fear:......", class_fear)
    print("neutral:...", class_neutral)
    print("surprise:..", class_surprise)

    # then we split the dataset in 75% based on the smallest number of sample
    train = []
    val = []

    smallest_dataset = int(rank[0] * percentage)

    angry = 0
    disgust = 0
    fear = 0
    happy = 0
    sad = 0
    surprise  = 0
    neutral  = 0

    for position in range (dataset.shape[0]):
        if(dataset[position, 0] == 0 and angry<smallest_dataset):
            angry += 1
            train.append(dataset[position])
        if(dataset[position, 0] == 0 and angry>=smallest_dataset):
            angry += 1
            val.append(dataset[position])
            
        if(dataset[position, 0] == 1 and disgust<smallest_dataset):
            disgust += 1
            train.append(dataset[position])
        if(dataset[position, 0] == 0 and disgust>=smallest_dataset):
            disgust += 1
            val.append(dataset[position])
            
        if(dataset[position, 0] == 2 and fear<smallest_dataset):
            fear += 1
            train.append(dataset[position])
        if(dataset[position, 0] == 2 and fear>=smallest_dataset):
            fear += 1
            val.append(dataset[position])
            
        if(dataset[position, 0] == 3 and happy<smallest_dataset):
            happy += 1
            train.append(dataset[position])
        if(dataset[position, 0] == 3 and happy>=smallest_dataset):
            happy += 1
            val.append(dataset[position])
            
        if(dataset[position, 0] == 4 and sad<smallest_dataset):
            sad += 1
            train.append(dataset[position])
        if(dataset[position, 0] == 4 and sad>=smallest_dataset):
            sad += 1
            val.append(dataset[position])
            
        if(dataset[position, 0] == 5 and surprise<smallest_dataset):
            surprise += 1
            train.append(dataset[position])
        if(dataset[position, 0] == 5 and surprise>=smallest_dataset):
            surprise += 1
            val.append(dataset[position])
            
        if(dataset[position, 0] == 6 and neutral<smallest_dataset):
            neutral += 1
            train.append(dataset[position])
        if(dataset[position, 0] == 6 and neutral>=smallest_dataset):
            neutral += 1
            val.append(dataset[position])
    
    train = numpy.array(train)
    val = numpy.array(val)
    print("train balanced:", train.shape[0])
    print("validation set:", val.shape[0])

    return (train, val)

dataset = import_csv('fer2018/fer2018.csv')
train, val = balance_dataset(dataset, 0.75)
train = pandas.DataFrame(train, columns=["y_train", "x_train"])
val = pandas.DataFrame(val, columns=["y_val", "x_val"])

train.to_csv("balanced_train_75percent.csv", encoding='utf-8', index=False)
val.to_csv("balanced_validation_75percent.csv", encoding='utf-8', index=False)

train, _ = balance_dataset(dataset, 1.)
train = pandas.DataFrame(train, columns=["y_train", "x_train"])

train.to_csv("balanced_100percent.csv", encoding='utf-8', index=False)
