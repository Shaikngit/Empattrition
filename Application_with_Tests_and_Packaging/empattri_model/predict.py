import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from empattri_model import __version__ as _version
from empattri_model.config.core import config
from empattri_model.processing.data_manager import load_pipeline
from empattri_model.processing.data_manager import pre_pipeline_preparation
from empattri_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
Employeeattrition_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
   
    validated_data = validated_data.reindex(columns = config.model_config.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
    
    if not errors:
        
        predictions = Employeeattrition_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    data_in = {
        'age': [41],
        'businesstravel': ['Travel_Rarely'],
        'dailyrate': [1102],
        'department': ['Sales'],
        'distancefromhome': [1],
        'education': [2],
        'educationfield': ['Life Sciences'],
        'employeecount': [1],
        'employeenumber': [1],
        'environmentsatisfaction': [2],
        'gender': ['Female'],
        'hourlyrate': [94],
        'jobinvolvement': [3],
        'joblevel': [2],
        'jobrole': ['Sales Executive'],
        'jobsatisfaction': [4],
        'maritalstatus': ['Single'],
        'monthlyincome': [5993],
        'monthlyrate': [19479],
        'numcompaniesworked': [8],
        'over18': ['Y'],
        'overtime': ['Yes'],
        'percentsalaryhike': [11],
        'performancerating': [3],
        'relationshipsatisfaction': [1],
        'standardhours': [80],
        'stockoptionlevel': [0],
        'totalworkingyears': [8],
        'trainingtimeslastyear': [0],
        'worklifebalance': [1],
        'yearsatcompany': [6],
        'yearsincurrentrole': [4],
        'yearssincelastpromotion': [0],
        'yearswithcurrmanager': [5]
    }




    make_prediction(input_data = data_in)