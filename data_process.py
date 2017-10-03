import numpy as np

def get_data(path_train, path_valid, path_test):
    TIME_STEP = 20
    train = np.squeeze(np.load(path_train)['arr_0']) # [20x10000, 64, 64]
    valid = np.squeeze(np.load(path_valid)['arr_0']) # [20x2000, 64, 64]
    test  = np.squeeze(np.load(path_test)['arr_0'])  # [20x3000, 64, 64]
    
    train = train.reshape((10000, TIME_STEP, 64, 64)).astype(float)/255
    train = np.transpose(train, (1, 0, 2, 3)) # [time_step, batch_size, 64, 64]
    train = np.expand_dims(train, axis=4)
    valid = valid.reshape((2000, TIME_STEP, 64, 64)).astype(float)/255
    valid = np.transpose(valid, (1, 0, 2, 3)) # [time_step, batch_size, 64, 64]
    valid = np.expand_dims(valid, axis=4)
    test  = test.reshape((3000, TIME_STEP, 64, 64)).astype(float)/255
    test  = np.transpose(test, (1, 0, 2, 3)) # [time_step, batch_size, 64, 64]
    test = np.expand_dims(test, axis=4)
    """
    train_list = []
    valid_list = []
    test_list  = []
    for idx in xrange(0, 4):
        for jdx in xrange(0, 4):
            train_list.append(train[:, :, idx*16:(idx+1)*16, jdx*16:(jdx+1)*16])
            valid_list.append(valid[:, :, idx*16:(idx+1)*16, jdx*16:(jdx+1)*16])
            test_list.append(test[:, :, idx*16:(idx+1)*16, jdx*16:(jdx+1)*16])
    train = np.asarray(train_list)
    train = np.transpose(train, (1, 2, 3, 4, 0))
    valid = np.asarray(valid_list)
    valid = np.transpose(valid, (1, 2, 3, 4, 0))
    test = np.asarray(test_list)
    test = np.transpose(test, (1, 2, 3, 4, 0))
    """
    return train, valid, test

def img_restore(img, time_step, batch_size):
    """
    img: [time_step, batch_size, 16, 16, 16]
    out: [time_step, batch_size, 64, 64]
    """
    out = np.zeros((time_step, batch_size, 64, 64))
    for idx in xrange(0, 4):
        for jdx in xrange(0, 4):
            out[:, :, idx*16:(idx+1)*16, jdx*16:(jdx+1)*16] = img[:, :, :, :, idx*4+jdx]
    
    return out