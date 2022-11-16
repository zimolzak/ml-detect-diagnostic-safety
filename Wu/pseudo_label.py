import numpy as np
import pandas as pd
import umap



class PseudoLabeler:
    def label(X_train, y_train, X_unlabeled):
        """
        :param y_train: n x 1 dimension label vector
        :return: n x 2 dimension label vector where column 1 is label and column 2 is probability/confidence
        """
        raise NotImplementedError()

        


def pseudo_label_dataset(labeler: PseudoLabeler, X_train, y_train, X_unlabeled, cutoff=0.66):
    pX, confidence, label_cols = labeler.label(X_train, y_train, X_unlabeled, cutoff)
    n = pX.shape[0]
    X = None
    y = None
    w = None
    for i in range(len(label_cols)):
        yslice = np.array([label_cols[i]] * n)
        wslice = confidence[:, i].copy()
        if X is None:
            X = pX.copy()
            y = yslice
            w = wslice
        else:
            X = pd.concat([X, pX], axis=0)
            y = np.concatenate((y, yslice), axis=0)
            w = np.concatenate((w, wslice), axis=0)
    X.reset_index(inplace=True, drop=True)
    return X, y, w

