import numpy as np
import pandas as pd

class confusion_matrix_metrics:
    
    # init
    def __init__(self, cm=None):
        if type(cm) is np.ndarray:
            if cm.ndim == 2:
                if cm.shape[0] == cm.shape[1]:
                    self.cm = cm
                else:
                    print('NOT A SQUARE ARRAY')
            else:
                print('NOT A 2D ARRAY')
        else:
            print('NOT AN ARRAY')
            
        self.prediction_classes = list(range(self.cm.shape[0]))
        self.condition_classes = list(range(self.cm.shape[1]))
        
    # calculate how many times a class of prediction was made
    def calc_prediction_amount(self):
        prediction_amounts = []
        agg = []
        
        for prediction in range(self.cm.shape[0]):
            agg.append([])
            
            for condition in range(self.cm.shape[1]):
                agg[prediction].append(self.cm[prediction, condition])
                
        prediction_amounts = [sum(c) for c in agg]
        
        return np.array(prediction_amounts)
    
    # calculate how many times a prediction was made of a class of condition (true or false)
    def calc_condition_amount(self):
        condition_amounts = []
        agg = []
        
        for condition in range(self.cm.shape[1]):
            agg.append([])
            
            for prediction in range(self.cm.shape[0]):
                agg[condition].append(self.cm[prediction, condition])
                
        condition_amounts = [sum(p) for p in agg]
        
        return np.array(condition_amounts)       
    
    # calculate how many times each class of prediction was true
    def calc_true_predictions(self):
        true_predictions = []
        
        for prediction in range(self.cm.shape[0]):
            for condition in range(self.cm.shape[1]):
                if prediction == condition:
                    
                    true_predictions.append(self.cm[prediction, condition])
                    
        return np.array(true_predictions)
    
    # calculate how many times each class of prediction was false
    def calc_false_predictions(self):
        false_predictions = []
        false_predictions_individual = []
        
        for prediction in range(self.cm.shape[0]):
            false_predictions_individual.append([])
            
            for condition in range(self.cm.shape[1]):
                if prediction != condition:
                    
                    false_predictions_individual[prediction].append(self.cm[prediction, condition])
                    
        false_predictions = [sum(c) for c in false_predictions_individual]
        
        return np.array(false_predictions)
    
    # calculate how many times each condition was correctly predicted
    def calc_predicted_conditions(self):
        predicted_conditions = []
        
        for condition in range(self.cm.shape[1]):
            for prediction in range(self.cm.shape[0]):
                if prediction == condition:
                    
                    predicted_conditions.append(self.cm[prediction, condition])
                    
        return np.array(predicted_conditions)
    
    # calculate how many times a condition was not predicted
    def calc_unpredicted_conditions(self):
        unpredicted_conditions = []
        unpredicted_conditions_individual = []
        
        for condition in range(self.cm.shape[1]):
            unpredicted_conditions_individual.append([])
            for prediction in range(self.cm.shape[0]):
                if prediction != condition:
                    
                    unpredicted_conditions_individual[condition].append(self.cm[prediction, condition])
                    
        unpredicted_conditions = [sum(p) for p in unpredicted_conditions_individual]
                    
        return np.array(unpredicted_conditions)
    
    # calculate PPV
    def calc_ppv(self):
        ppv_list = []
        
        for prediction in range(self.cm.shape[0]):
            ppv_list.append(
                self.get_true_predictions(prediction)/
                self.get_prediction_amount(prediction)
            )
        
        return np.array(ppv_list)
    
    # calculate NPV
    def calc_npv(self):
        npv_list = []
        
        for prediction in range(self.cm.shape[0]):
            non_pred_preds = [e for e in self.prediction_classes if e!=prediction]
            
            npv_list.append(
                sum(self.get_true_predictions()[non_pred_preds])/
                (sum(self.get_prediction_amount()[non_pred_preds]))
            )
        
        return np.array(npv_list)
    
    # calculate sensitivity
    def calc_sensitivity(self):
        specificity_list = []
        
        for condition in range(self.cm.shape[1]):
            specificity_list.append(
                self.get_predicted_conditions(condition)/
                self.get_condition_amount(condition)
            )
        
        return np.array(specificity_list)
    
    # calculate specificity
    def calc_specificity(self):
        sensitivity_list = []
        
        for condition in range(self.cm.shape[1]):
            non_cond_conds = [e for e in self.condition_classes if e!=condition]
            
            sensitivity_list.append(
                sum(self.get_predicted_conditions()[non_cond_conds])/
                sum(self.get_condition_amount()[non_cond_conds])
            )
        
        return np.array(sensitivity_list)
    
    # getter for calc_prediction_amount
    def get_prediction_amount(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_prediction_amount(), nan=0)
        else:
            return np.nan_to_num(self.calc_prediction_amount()[prediction_index], nan=0)
    
    # getter for calc_condition_amount
    def get_condition_amount(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_condition_amount(), nan=0)
        else:
            return np.nan_to_num(self.calc_condition_amount()[prediction_index], nan=0)
    
    # getter for calc_true_predictions
    def get_true_predictions(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_true_predictions(), nan=0)
        else:
            return np.nan_to_num(self.calc_true_predictions()[prediction_index], nan=0)
    
    # getter for calc_false_predictions
    def get_false_predictions(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_false_predictions(), nan=0)
        else:
            return np.nan_to_num(self.calc_false_predictions()[prediction_index], nan=0)
    
    # getter for calc_predicted_conditions
    def get_predicted_conditions(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_predicted_conditions(), nan=0)
        else:
            return np.nan_to_num(self.calc_predicted_conditions()[prediction_index], nan=0)
    
    # getter for calc_unpredicted_conditions
    def get_unpredicted_conditions(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_unpredicted_conditions(), nan=0)
        else:
            return np.nan_to_num(self.calc_unpredicted_conditions()[prediction_index], nan=0)
        
    # getter for calc_ppv
    def get_ppv(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_ppv(), nan=0)
        else:
            return np.nan_to_num(self.calc_ppv()[prediction_index], nan=0)
    
    # getter for calc_npv
    def get_npv(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_npv(), nan=0)
        else:
            return np.nan_to_num(self.calc_npv()[prediction_index], nan=0)
    
    # getter for calc_sensitivity
    def get_sensitivity(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_sensitivity(), nan=0)
        else:
            return np.nan_to_num(self.calc_sensitivity()[prediction_index], nan=0)
    
    # getter for calc_specificity
    def get_specificity(self, prediction_index=-1):
        if prediction_index == -1:
            return np.nan_to_num(self.calc_specificity(), nan=0)
        else:
            return np.nan_to_num(self.calc_specificity()[prediction_index], nan=0)