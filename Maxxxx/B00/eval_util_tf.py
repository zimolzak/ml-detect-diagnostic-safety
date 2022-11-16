from eval_util import *
import tensorflow as tf
import pseudo_label
import cluster

"""

This file is for specific eval util functions that depend on the tensorflow package

"""


AUC = tf.keras.metrics.AUC()


def trainNN(X_train, y_train, epochs=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(X_train.columns),)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam",
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy', AUC])
    history = model.fit(X_train.to_numpy(dtype="float32"), y_train.to_numpy(dtype="float32"), epochs=epochs, verbose=0)
    
    print(model.evaluate(X_train.to_numpy(dtype="float32"), y_train.to_numpy(dtype="float32")))
    
    plt.plot(history.history[AUC.name])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.legend(['auc', 'acc', 'loss'], loc='upper left')
    plt.show()
    
    return model


def evaluateNN(model, X_test, y_test):
    model.evaluate(X_test.to_numpy(dtype="float32"), y_test.to_numpy(dtype="float32"))
    prediction = model.predict(X_test.to_numpy(dtype="float32"))
    result = sklearn.metrics.classification_report(y_test.to_numpy(), 
                                                   (prediction > 0.5).astype(int), output_dict=True)
    result["auc"] = sklearn.metrics.roc_auc_score(y_test.to_numpy(), prediction)
#     AUC.reset_states()
#     AUC.update_state(y_test.to_numpy(), prediction[:,0])
#     print(AUC.result())
#     AUC.reset_states()
    return result


def trainPNN(X_train, y_train, pseudo_labeler, unlabeled_df, epochs=200):
    pX, py, pw = pseudo_label.pseudo_label_dataset(pseudo_labeler, X_train, y_train, unlabeled_df)
    n = int(pX.shape[0]/2)
    X = pX.iloc[:n, :].to_numpy(dtype="float32")
    y = pw[:n].astype("float32")
    y = np.stack([1- y, y], -1)
    pmodel = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(pX.columns),)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    pmodel.compile(optimizer="adam", loss=tf.keras.losses.KLDivergence(), metrics=["accuracy"])
    history = pmodel.fit(X, y, epochs=epochs, verbose=0)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.legend(['acc', 'loss'], loc='upper left')
    plt.show()
    
    return pmodel

def evaluatePNN(model, X_test, y_test):
    nn_y_test = y_test.to_numpy(dtype="float32")
    nn_y_test = np.stack([1- nn_y_test, nn_y_test], -1)
    model.evaluate(X_test.to_numpy(dtype="float32"), nn_y_test)
    prediction = model.predict(X_test.to_numpy(dtype="float32"))
    result = sklearn.metrics.classification_report(y_test.to_numpy(), 
                                                   np.argmax(prediction, axis=1), output_dict=True)
    result["auc"] = sklearn.metrics.roc_auc_score(y_test.to_numpy(), np.argmax(prediction, axis=1))
    return result

def trainKNN(X_train, y_train, epochs=200, *, n_neighbors=3, c=1e-6):
    reducer = umap.UMAP(metric="correlation", min_dist=0, n_neighbors=5)
    knn = cluster.HeuristicKNNClusterClassifier(reducer, n_neighbors=n_neighbors, c=c)
    knn.fit(X_train, y_train)
    
    print(knn.score(X_train, y_train))

    return knn

def evaluateKNN(model, X_test, y_test):
    prediction = model.predict(X_test)
    result = sklearn.metrics.classification_report(y_test.to_numpy(), prediction, output_dict=True)
#     result["auc"] = sklearn.metrics.roc_auc_score(y_test.to_numpy(), prediction)
#     AUC.reset_states()
#     AUC.update_state(y_test.to_numpy(), prediction[:,0])
#     print(AUC.result())
#     AUC.reset_states()
    return result



# the following three functions are wrappers to unify the function signatures, so that they can be ran with runClassifierEvalCV
def evaluateNNwrapper():
    return lambda X_train, y_train, unlabeled_df, epochs=200: trainNN(X_train, y_train, epochs)

def evaluatePNNwrapper(pseudo_labeler):
    return lambda X_train, y_train, unlabeled_df, epochs=200: trainPNN(X_train, y_train, pseudo_labeler, unlabeled_df, epochs)

def evaluateKNNwrapper(n_neighbors=3, c=1e-6):
    return lambda X_train, y_train, unlabeled_df, epochs=200: trainPNN(X_train, y_train, epochs, n_neighbors=n_neighbors, c=c)


def runClassifierEvalCV(feature_vec, label_df, times, train_func, eval_func, kfold=5):
    models = []
    results = []
    
    for i in range(times):
        labeled_df, unlabeled_df, X_cols, y_col = util.makeDataset(feature_vec, label_df)
    
        skf = sklearn.model_selection.StratifiedKFold(n_splits=kfold, shuffle=True, random_state=RANDOM_STATE_SEED + i)
    
        for train_index, test_index in skf.split(labeled_df[X_cols], labeled_df[y_col]):
            X_train, X_test = labeled_df[X_cols].iloc[train_index], labeled_df[X_cols].iloc[test_index]
            y_train, y_test = labeled_df[y_col].iloc[train_index], labeled_df[y_col].iloc[test_index]

            model = train_func(X_train, y_train, unlabeled_df[X_cols])
            models.append(model)
            results.append(eval_func(model, X_test, y_test))
            
    return models, results