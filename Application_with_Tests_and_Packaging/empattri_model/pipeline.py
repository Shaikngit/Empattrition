import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from empattri_model.config.core import config
#from empattri_model.processing.features import SingleColumnNameChanger
from empattri_model.processing.features import OutlierHandler, CategoricalEncoder  

Employeeattrition_pipe = Pipeline([

    ######## Handle outliers ########
    ('handle_outliers_monthlyincome', OutlierHandler(config.model_config.monthlyincome_var)),
    ('handle_outliers_numcompaniesworked', OutlierHandler(config.model_config.numcompaniesworked_var)),
    ('handle_outliers_stockoptionlevel', OutlierHandler(config.model_config.stockoptionlevel_var)),
    ('handle_outliers_performancerating', OutlierHandler(config.model_config.performancerating_var)),
    ('handle_outliers_totalworkingyears', OutlierHandler(config.model_config.totalworkingyears_var)),
    ('handle_outliers_trainingtimeslastyear', OutlierHandler(config.model_config.trainingtimeslastyear_var)),
    ('handle_outliers_yearsatcompany', OutlierHandler(config.model_config.yearsatcompany_var)),
    ('handle_outliers_yearsincurrentrole', OutlierHandler(config.model_config.yearsincurrentrole_var)),
    ('handle_outliers_yearssincelastpromotion', OutlierHandler(config.model_config.yearssincelastpromotion_var)),
    ('handle_outliers_yearswithcurrmanager', OutlierHandler(config.model_config.yearswithcurrmanager_var)),

    
    ######## Categorical encoders ########
    ('encode_businesstravel', CategoricalEncoder(config.model_config.businesstravel_var)),
    ('encode_department', CategoricalEncoder(config.model_config.department_var)),
    ('encode_educationfield', CategoricalEncoder(config.model_config.educationfield_var)),
    ('encode_gender', CategoricalEncoder(config.model_config.gender_var)),
    ('encode_jobrole', CategoricalEncoder(config.model_config.jobrole_var)),
    ('encode_maritalstatus', CategoricalEncoder(config.model_config.maritalstatus_var)),
    ('encode_overtime', CategoricalEncoder(config.model_config.overtime_var)),


    
    # Mapper 

    # ('map_attri', Mapper('attrition', mappings = config.model_config.attr_mappings)),
    # ('target_encoder', TargetEncoder()),

    # Scale features
    # ('scaler', StandardScaler()),

    # Classifier
    ('model_rf', CatBoostClassifier(learning_rate = config.model_config.learning_rate,
                            depth = config.model_config.depth,
                            scale_pos_weight = config.model_config.scale_pos_weight,
                            l2_leaf_reg = config.model_config.leaf_reg,
                            border_count = config.model_config.border_count,verbose=False))
])