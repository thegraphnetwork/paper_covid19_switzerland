import pandas as pd
from datetime import timedelta 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ngboost.scores import LogScore
from ngboost.distns import LogNormal
from ngboost.learners import default_tree_learner
import scipy.cluster.hierarchy as hcluster
from epigraphhub.analysis.forecast_models.plots import plot_val
from epigraphhub.analysis.forecast_models.ngboost_models import NGBModel
from epigraphhub.analysis.forecast_models.metrics import compute_metrics
from sklearn.metrics import (mean_absolute_error as mae, mean_squared_error as mse, mean_squared_log_error as msle,
                             mean_absolute_percentage_error as mape)
                             

params_model = {
    "Base": default_tree_learner,
    "Dist": LogNormal,
    "Score": LogScore,
    "natural_gradient": True,
    "verbose": False,
    "col_sample": 0.9,
    "n_estimators": 100,
    "learning_rate": 0.05,
}

def remove_zeros(tgt):
    """
    Function to remove the zeros of the target curve. It needs to be done to us be able
    to use the LogNormal dist.
    :params tgt: array.
    """

    tgt[tgt == 0] = 0.01

    return tgt

def train_eval_article(
    target_curve_name,
    canton,
    ini_date="2020-03-01",
    end_train_date=None,
    end_date=None,
    ratio=0.75,
    ratio_val=0.15,
    early_stop=5,
    params_model=params_model,
    predict_n=14,
    look_back=14,
):

    """
    This function it was create to allow the reproduction of the results of the article.
    Function to train and evaluate the model for one georegion.

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params canton: canton of interest
    :params ini_date: string. Determines the beggining of the train dataset
    :params end_train_date: string. Determines the beggining of end of train dataset. If end_train_date
                           is not None, then ratio isn't used.
    :params end_date: string. Determines the end of the dataset used in validation.
    :params ratio: float. Determines which percentage of the data will be used to train the model
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.
    :params predict_n: int. Number of days that will be predicted.
    :params look_back: int. Number of the last days that will be used to forecast the next days.

    returns: Dataframe.
    """

    target_name = f"{target_curve_name}_{canton}"

    df = pd.read_csv(f"data_article/data_{canton}.csv")

    df.set_index("datum", inplace=True)

    df.index = pd.to_datetime(df.index)

    df = df.fillna(0)

    df[target_name] = remove_zeros(df[target_name].values)

    if any(df[target_name] > 1):
        
        m = NGBModel(look_back = look_back,
            predict_n = predict_n, 
            validation_split = ratio_val, 
            early_stop = early_stop, params_model = params_model)

        df_pred = m.train_eval(
            target_name,
            df,
            ini_date=ini_date,
            end_train_date=end_train_date,
            end_date=end_date,
            ratio=ratio)

        df_pred["canton"] = [target_name[-2:]] * len(df_pred)

    else:
        print('erro')
        df_pred = pd.DataFrame()
        df_pred["target"] = df[target_name]
        df_pred["date"] = 0
        df_pred["lower"] = 0
        df_pred["median"] = 0
        df_pred["upper"] = 0
        df_pred["train_size"] = 0
        df_pred["canton"] = target_name[-2:]

    return df_pred

def train_eval_metrics(target_curve_name = 'hosp', predictors = ['foph_test_d', 'foph_cases_d', 'foph_hosp_d'], ini_date = '2020-05-01', ratio = 0.8,  early_stop = 10, end_date = '2022-04-30',  canton = 'GE'):
    
    df_eval = train_eval_article(target_curve_name, canton,ini_date = ini_date, end_date = end_date)
    
    dict_name = {'hosp': 'New hospitalizations',
                'total_hosp': 'Total Hospitalizations', 
                'icu_patients': 'Total ICU patients'}
    
    plot_val(df_eval, title = f'', path = 'plots', name = f'val_{target_curve_name}_{canton}', save = True)

    df_m = compute_metrics(df_eval)
    
    df_m.to_csv(f'metrics/val_{target_curve_name}_{canton}.csv')
    
    return df_eval, df_m


def train_article(
    target_curve_name,
    canton,
    path="../opt/models/saved_models/ml",
    ini_date="2020-03-01",
    end_date=None,
    parameters_model=params_model,
    predict_n=14,
    look_back=14,
):

    """
    This function it was create to allow the reproduction of the results of the article.
    Function to train the model for one georegion

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params ini_date: Determines the beggining of the train dataset
    :params path: Determines  where the model trained will be saved
    :params update_data: Determines if the data from the Geneva hospital will be used.
                        this params only is used when canton = GE and target_curve_name = hosp.
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.
    :params predict_n: int. Number of days that will be predicted.
    :params look_back: int. Number of the last days that will be used to forecast the next days.

    :returns: None
    """

    target_name = f"{target_curve_name}_{canton}"

    # getting the data
    df = pd.read_csv(f"data_article/data_{canton}.csv")

    df.set_index("datum", inplace=True)

    df.index = pd.to_datetime(df.index)

    df = df.fillna(0)

    df[target_name] = remove_zeros(df[target_name].values)

    if any(df[target_name] > 1):
        
        m = NGBModel(look_back = look_back,
            predict_n = predict_n, 
            validation_split = 0.15, 
            early_stop = 5, params_model = params_model)
        

        m.train(
            target_name,
            df,
            ini_date=ini_date,
            path=path,
            end_date=end_date
        )

    else:
        print(
            f"The model to forecast {target_name} was not trained, since the series has no value bigger than one."
        )

    return

def forecast_article(
    target_curve_name,
    canton,
    end_date=None,
    path="../opt/models/saved_models/ml",
    predict_n=14,
    look_back=14,
):
    """
    This function it was create to allow the reproduction of the results of the article.
    Function to make the forecast for one canton, it will load pre trained models.

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: string to indicate the interest canton
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params end_date: string. Determines from what day the forecast will be computed.
    :params path: string. Indicates where the models trained are saved.
    :params predict_n: int. Number of days that will be predicted.
    :params look_back: int. Number of the last days that will be used to forecast the next days.

    returns: Dataframe with the forecast for all the cantons
    """

    target_name = f"{target_curve_name}_{canton}"

    df = pd.read_csv(f"data_article/data_{canton}.csv")

    df.set_index("datum", inplace=True)

    df.index = pd.to_datetime(df.index)

    df = df.fillna(0)
    
    m = NGBModel(look_back = look_back,
            predict_n = predict_n, 
            validation_split = 0.15, 
            early_stop = 5, params_model = params_model)

    df_for = m.forecast(
        df,
        end_date=end_date,
        path=path,
    )

    return df_for

def plot_forecast(target_name, canton, df_for):
    
    dict_name = {'hosp': ['new hospitalizations', 'datum', 'entries', 'foph_hosp_d'],
                'total_hosp': ['Total Hospitalizations', 'date', 'total_covid19patients', 'foph_hospcapacity_d'], 
                'icu_patients': ['Total ICU patients', 'date', 'icu_covid19patients', 'foph_hospcapacity_d']}
    
    title = f'Forecast {dict_name[target_name][0]} in {canton}'
    
    df = pd.read_csv(f'data_article/data_{canton}.csv')
    
    df.set_index("datum", inplace = True)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    
    df = df.resample("D").mean()
    
    plt.figure()
    
    plt.plot(df.loc[df_for.index][f'{target_name}_{canton}'], label = 'Data', color = 'black')
    
    plt.plot(df_for['median'], label = 'Median', color = 'tab:orange')
    
    plt.fill_between(df_for.index, df_for.upper, df_for.lower, color = 'tab:orange', alpha = 0.5)
    
    plt.xlabel('date')
    plt.ylabel('incidence')
    
    #plt.title(title)
    plt.xticks(rotation=25)
    plt.legend()
    plt.grid()
    plt.savefig(f'plots/forecast_{target_name}_{canton}.png', bbox_inches='tight')
    plt.show()
    
    # computing some metrics 
    df_metrics = pd.DataFrame(columns = ['metrics', 'forecast_error'] )
    
    metrics = ['mean_absolute_error',
               'mean_squared_error', 'root_mean_squared_error', 'mean_squared_log_error',
                'mean_absolute_percentage_error']
    
    df_metrics['metrics'] = metrics 
    
    y_true = df.loc[df_for.index][f'{target_name}_{canton}']
    
    y_for = df_for['median']
    
    df_metrics['forecast_error'] = [ 
                                mae(y_true, y_for),
                                mse(y_true, y_for),
                                mse(y_true, y_for, squared = False),
                                msle(y_true, y_for),
                                mape(y_true, y_for) ]
    
    df_metrics.to_csv(f'metrics/for_metrics_{target_name}_{canton}.csv')
    
    return df_metrics     


def train_forecast(target_curve_name = 'hosp', canton = 'GE',
                    path = 'saved_models'):
    
    train_article(
    target_curve_name,
    canton,
    path = path,
    ini_date="2020-05-01",
    end_date = '2022-05-01')
    
    df_for = forecast_article(
    target_curve_name,
    canton,
    end_date = '2022-05-01', 
    path=path)
    
    df_m = plot_forecast(target_curve_name, canton, df_for)
    
    df_m.set_index('metrics', inplace = True)
    
    return df_for, df_m  
