import numpy as np

def percent_overlap(y_true, y_pred):
    
    """
    Calculate the amount of lesion that is being predicted compare to the true mask of the lesion
    """
    score = (np.sum(y_true * y_pred))/ (np.sum((y_true)))
    return score



def IOU(y_true, y_pred):
    """
    Calculate the Intersection of Union of the images
    """

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    score = intersection / union

    return score

def dice_coefficient (y_true, y_pred):
    """
    Calculate the dice coeff of the model
    """
    coeff = (2 * np.sum(y_true * y_pred))/ (np.sum(y_true) + np.sum(y_pred))
    return coeff

