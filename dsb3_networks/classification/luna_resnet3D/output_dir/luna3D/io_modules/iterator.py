'''
Iterator base Class for List and Proto Iterators
'''

class Iterator(object):
    def __init__(self):
        '''
        Default Constructor
        '''
        pass
      
    def initialize(self):
        '''
        initialize reader (queue runners)
        '''
        pass
    
    def need_queue_runners(self):
        '''
        Return true if there are any queue runners 
        that have to be started and stoped
        '''
        return False

    def read_batch(self):
        '''
        Function to read list/record
        '''
        pass

    def get_data_batch(self):
        '''
        Function returns the next list of numpy labels for a feed_dict
        '''
        return None 

    def get_label_batch(self):
        '''
        Function returns the next list of numpy data for a feed_dict
        '''
        return None

    def data_batch(self):
        '''
        Function returns a list of data-tensors/placeholders
        '''
        pass
    
    def label_batch(self):
        '''
        Function returns a list of label-tensors/placeholders
        '''
        pass
    

