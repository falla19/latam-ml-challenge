import pandas as pd
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import numpy as np
import pickle

from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        """
            Initialize model by loading it from disk if it exists, otherwise, set it to None.
            
            DelayModel uses the following attributes:
                - self._model: model to be used for prediction.
                - self.top_10_features: top 10 features to be used for preprocessing and prediction.
                - self.THRESHOLD_IN_MINUTES: threshold in minutes to be used for preprocessing.
        """
        try:
            self._model = pickle.load(open("data/model.h5", 'rb'))
        except:
            self._model = None # Model should be saved in this attribute.
        self.top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        self.THRESHOLD_IN_MINUTES = 15
        
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.
        
        The function goes through the following steps:
            - Create a new column called period_day based on the scheduled date and time of the flight. 
                * This column contains the period of the day in which the flight is scheduled to depart (mañana, tarde, noche).
            - Create a new column called high_season based on the scheduled date and time of the flight.
                * This column contains 1 if the flight is scheduled to depart in high season (15-Dec - 31-Dec, 1-Jan - 3-Mar, 15-Jul - 31-Jul, 11-Sep - 30-Sep) and 0 otherwise.
            - Create a new column called min_diff based on the difference between the scheduled time and the actual time.
                * This column contains the difference between the scheduled time and the actual time in minutes.
            - Create a new column called delay based on the difference between the scheduled time and the actual time.
                * This column contains 1 if the difference between the scheduled time and the actual time is greater than 15 minutes and 0 otherwise.
            - Create dummy variables for the following columns: OPERA, TIPOVUELO, MES.
                * The dummy variables are created for the top 10 values of each column.
            - Save the features to be used for training.
            - Save the model to be used for prediction.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
            
        data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
        
        data['delay'] = np.where(data['min_diff'] > self.THRESHOLD_IN_MINUTES, 1, 0)
            
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
            
        pickle.dump(features, open("data/all_features.p", 'wb'))
        
        return (features[self.top_10_features], data['delay'].to_frame()) if target_column else features[self.top_10_features]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.
        
        The function goes through the following steps:
            - Find the number of samples for each class to balance the data.
            - Instantiate a LogisticRegression model with the class_weight parameter set to the balanced weights.
            - Fit the model with the preprocessed data.
            - Save the model to be used for prediction.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        
        n_y0 = len(target[target.iloc[:,0] == 0])
        n_y1 = len(target[target.iloc[:,0] == 1])
        
        self._model = LogisticRegression(class_weight={1: n_y0/len(target), 0: n_y1/len(target)})
        self._model.fit(features, target.to_numpy().ravel())
        
        pickle.dump(self._model, open("data/model.h5", 'wb'))
        
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.
        
        The function goes through the following steps:
            - Predict the target for the preprocessed data.
            - Return the predicted targets.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        
        predicted_target = self._model.predict(features)
        
        return predicted_target.tolist()
    
    def get_period_day(self, date) -> str:
        """
        Get period of the day based on the scheduled date and time of the flight.
        
        The function goes through the following steps:
            - Convert the scheduled date and time of the flight to a datetime object.
            - Compare the scheduled date and time of the flight with the following ranges:
                * mañana: 05:00 - 11:59
                * tarde: 12:00 - 18:59
                * noche: 19:00 - 23:59, 00:00 - 4:59
            - Return the period of the day in which the flight is scheduled to depart (mañana, tarde, noche).

        Args:
            date (str): scheduled date and time of the flight. 

        Returns:
            str: period of the day in which the flight is scheduled to depart (mañana, tarde, noche).
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
        
        if(date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif(
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return 'noche'
        
    def is_high_season(self, fecha) -> int:
        """
        Check if the flight is scheduled to depart in high season.
        
        The function goes through the following steps:
            - Convert the scheduled date and time of the flight to a datetime object.
            - Compare the scheduled date and time of the flight with the following ranges:
                * high season: 15-Dec - 31-Dec, 1-Jan - 3-Mar, 15-Jul - 31-Jul, 11-Sep - 30-Sep
            - Return 1 if the flight is scheduled to depart in high season and 0 otherwise.
        
        Args:
            fecha (str): scheduled date and time of the flight.
            
        Returns:
            int: 1 if the flight is scheduled to depart in high season and 0 otherwise.
        """
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
        
    def get_min_diff(self, data) -> float:
        """
        Get the difference between the scheduled time and the actual time in minutes.
        
        The function goes through the following steps:
            - Convert the scheduled date and time of the flight to a datetime object.
            - Convert the actual date and time of the flight to a datetime object.
            - Calculate the difference between the scheduled time and the actual time in minutes.
            - Return the difference between the scheduled time and the actual time in minutes.
            
        Args:
            data (pd.DataFrame): row of the dataframe.
            
        Returns:
            float: difference between the scheduled time and the actual time in minutes.
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
    
    def check_response(self, data: object) -> Union[Tuple[str, str], List[int]]:
        """
        Check if the data has correct values in their columns. If the data has correct values in their columns, predict the delay of the flights in the data, otherwise, return an error to raise an exception.
        
        the function goes through the following steps:
            - Load the features to be used for prediction.
            - Check if the data has correct values in their columns.
            - Return the predicted delay of the flights in the data or an error to raise an exception.
            
        Args:
            data (object): data to predict.
            
        Returns:
            Tuple[str, str]: error to raise an exception.
            or
            List[int]: predicted delay of the flights in the data. 
        """
        
        features = pickle.load(open("data/all_features.p", 'rb'))
        
        opera_columns = [col for col in features.columns.tolist()
                        if col.startswith('OPERA')]
        replace_opera = [col.replace('OPERA_', '') for col in opera_columns]
        
        data_df = []

        for flight in data.flights:
            if (flight['MES'] > 12 or flight['MES'] < 1):
                return ('MES', "Column MES not found")
            elif not (flight['TIPOVUELO'] == 'N' or flight['TIPOVUELO'] == 'I'):
                return ('TIPOVUELO', "Column TIPOVUELO not found")
            elif (flight['OPERA'] not in replace_opera):
                return ('OPERA', "Column OPERA not found")
            else:
                data_df.append(flight)
                
        data_df = pd.DataFrame(data_df, columns=['OPERA', 'TIPOVUELO', 'MES'])
    
        features_df = pd.concat([
            pd.get_dummies(data_df['OPERA'], prefix='OPERA'),
            pd.get_dummies(data_df['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data_df['MES'], prefix='MES')],
            axis=1
        )
        
        data_predict = [[0] * len(self.top_10_features)]
        
        for col in features_df.columns.tolist():
            if col in self.top_10_features:
                data_predict[0][self.top_10_features.index(col)] = features_df[col].values[0]
        
        df = pd.DataFrame(data_predict, columns=self.top_10_features)
        
        
        return self.predict(df)