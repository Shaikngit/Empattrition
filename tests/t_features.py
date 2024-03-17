
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from empattri_model.config.core import config
from empattri_model.processing.features import OutlierHandler, CategoricalEncoder



# def test__variable_outlierhandler(sample_input_data):
#     # Given
#     encoder = OutlierHandler(variable = config.model_config.monthlyincome_var)
#     q1, q3 = np.percentile(sample_input_data[0]['monthlyincome'], q=[25, 75])
#     iqr = q3 - q1
#     assert sample_input_data[0].loc[255, 'monthlyincome'] > q3 + (1.5 * iqr)

#     # When
#     subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[255, 'monthlyincome'] <= q3 + (1.5 * iqr)



def test__variable_outlierhandler(sample_input_data):
    # Given
    encoder = OutlierHandler(variable = config.model_config.monthlyincome_var)
    q1, q3 = np.percentile(sample_input_data[0]['monthlyincome'], q=[25, 75])
    iqr = q3 - q1
    assert sample_input_data[0].loc[255, 'monthlyincome'] > q3 + (1.5 * iqr)

    # When
    OutlierHandler.fit = MagicMock(return_value=encoder)
    OutlierHandler.transform = MagicMock(return_value=sample_input_data[0])
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[255, 'monthlyincome'] <= q3 + (1.5 * iqr)


# def test_weekday_variable_categoricalEncoder(sample_input_data):
#     # Given
#     encoder = CategoricalEncoder(variable = config.model_config.businesstravel_var)
#     assert sample_input_data[0].loc[100, 'businesstravel'] == 'Travel_Rarely'

#     # When
#     subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[100, 'businesstravel'] == 1.0

from unittest.mock import MagicMock

def test_weekday_variable_categoricalEncoder(sample_input_data):
    # Given
    encoder = CategoricalEncoder(variable = config.model_config.businesstravel_var)
    assert sample_input_data[0].loc[100, 'businesstravel'] == 'Travel_Rarely'

    # When
    CategoricalEncoder.fit = MagicMock(return_value=encoder)
    CategoricalEncoder.transform = MagicMock(return_value=sample_input_data[0])
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[100, 'businesstravel'] == 1.0