
import time

import pandas
import sklearn
import numpy


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

    #replacing a long string to a numpy array
    for line in range(data_frame.shape[0]):
        pixels = data[line, 1].split(" ")
        
        pixels = numpy.array(pixels, dtype=numpy.uint8)
        data[line, 1] = pixels

    print("\nimportation done in:", time.time() - start_time)
    return data

def split_dataset(number_subset, dataset):
    """take a dataframe and split it in x number of subset for cross validation
    
    Arguments:
        number_subset {int} -- number of subsets
        dataset {numpy.array} -- [description]
    
    Returns:
        list of subsets
        the list of sets is segmented as following:
        [(x_train, y_train, x_val, y_val), ...]
    """
    # we split our dataset in a list of small sets
    small_batch =  numpy.array_split(dataset, number_subset)
    # each of the element of this list is a dataset: small_batch[0] is the subset 0 with the X and Y

    set_list = []
    # for each subset we want to create
    for set_number in range(number_subset):
        x_train = []
        y_train = []
        x_validation = []
        y_validation = []

        # we will add every small set created earlier and use the set at [position] as validation set
        """
            batch a
            batch b
            batch c
            batch d

            subset 1 = b + c + d
            subset 2 = a + c + d
            subset 3 = a + b + d
            subset 4 = a + b + c
        """

        for position in range(number_subset):

            if(position == set_number):
                x_validation.append(small_batch[position][:, 1].tolist())
                y_validation.append(small_batch[position][:, 0].tolist())
                import ipdb; ipdb.set_trace()

            if(position != set_number):
                x_train.append(small_batch[position][:, 1].tolist())
                y_train.append(small_batch[position][:, 0].tolist())
        
        set_list.append( 
            (numpy.concatenate(x_train), numpy.concatenate(y_train), numpy.concatenate(x_validation), numpy.concatenate(y_validation))
            )
    return set_list

def classic_split(dataset, percentage):
    """split a dataset only in two (no cross validation)
    
    Arguments:
        dataset {numpy array} -- 
        percentage {float} -- eg: 0.75 for 75%
    """

    size = dataset.shape[0]
    split_position = int(size * percentage)

    # we will take all images and put them in train set, when we arrive at 75% of the dataset, 
    #     the remaining images will be put in the validation set
    x_train = []
    y_train = []
    x_validation = []
    y_validation = []
    import ipdb; ipdb.set_trace()
    for position in range(size):
        # if we are below 75%
        if(position <= split_position):
            x_train.append(dataset[position, 1])
            y_train.append(dataset[position, 0])
        
        # if we added 75% of the dataset in train:
        if (position > split_position):
            x_validation.append(dataset[position, 1])
            y_validation.append(dataset[position, 0])

    x_train = numpy.array(x_train)
    y_train = numpy.array(y_train)
    x_validation = numpy.array(x_validation)
    y_validation = numpy.array(y_validation)

    return (x_train, y_train, x_validation, y_validation)
