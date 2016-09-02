import numpy as np
from passage.utils import save, load
from passage.layers import Embedding, GatedRecurrent, Dense, OneHot, LstmRecurrent, Generic
from passage.models import RNN
import random
import sys

#TODO defunct
def batch_predict_auto(model, data, max_batch_size):
    """
    @depricated
    batch_predict_auto

    Inputs:
        (same as batch_predict, plus:)
        max_batch_size - Maximum batch size to try before backing off

    Returns:
        (same as batch_predict)
    """
    bs = max_batch_size
    while bs > 0:
        try:
            return batch_predict(model, data, bs)
        except MemoryError:
            if bs > 0:
                print "Batch size", bs, "failed.",
                bs = bs / 2
                print "Trying", bs, "..."

    #If we get through the while loop without returning, we must
    #have exhausted every possible batch size and failed each time!
    print "ERROR - Not enough GPU memory for minimum batch size!"
    raise MemoryError("Out of GPU memory, long sequences")


def batch_predict(models, data, result = [], max_batch=64):
    """
    batch_predict -- For a given set of models, computes a set of predictions on the dataset.
        Acts as a generator.

    Inputs
        models - List of models to use for prediction. See passage.util.load.
        data - List of tuples that includes, (tokens, labels, names ...) and any other extra information.
            This information is returned along with the prediction results.
        max_batch - Maximum batch size to use at prediction time.
    
    Returns
        Returns a generator of model predictions.
        
    """
    #correct for a single model passed instead of a list
    if type(models) != list:
        models = [models]

    batches = _prepare_batches(data, max_batch=64)

    for b in batches:
        model_predictions = []
        dna = zip(*b)[0]

        for model in models:
            predictions =  model.predict(dna)
            model_predictions.append( predictions.tolist() + zip(*b)[1:] )

        yield model_predictions
        


def _prepare_batches(data, max_batch=64):
    """
    prepare_batches 
        Orders sequences by length, this is required for parallel prediction.

    Inputs 
        data - list of tokenized sequences to be used for prediction.
        max_batch - The maximum allowed batch size for prediction. Try decreasing this number[64]
    Returns  
    """
    length_dictionary = {}
    data = sorted(data, key=lambda tup: len(tup[0]), reverse=False)
    for d in data:
        length = len(d[0])
        if length not in length_dictionary:
            length_dictionary[length] = []
        length_dictionary[length].append(d)
    batches = []
    for length in length_dictionary:
        datas = length_dictionary[length]
        for b in range(0, len(datas), max_batch):
            batches.append(datas[b:b+max_batch])
    return batches
