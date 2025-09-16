# import _pickle as cPickle
import pickle


def store_in_pickle(file_address, data):
    try:
        p = pickle.Pickler(open(file_address,"wb"),  protocol=pickle.HIGHEST_PROTOCOL) 
        p.fast = False 
        p.dump(data)
        return True
    except Exception as error:
        print(error)
        return False
    
def retrieve_from_pickle(file_address):
    try:
        p = pickle.Unpickler(open(file_address,"rb")) 
        seqs_list = p.load()
        return seqs_list
    except:
        return None