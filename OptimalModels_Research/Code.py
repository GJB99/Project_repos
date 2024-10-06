# -*- coding: utf-8 -*-
"""
Created on Fri Apr 7 12:40:45 2023
@author: guusb
"""

from pandas_datareader import data as pdr
from arch import arch_model
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.dates as mdates
import warnings
import sys
#tests and other metrics than loglikelihood and backtesting
from statsmodels.tsa.stattools import adfuller, acf
from arch.unitroot import PhillipsPerron
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from epftoolbox.evaluation import plot_multivariate_DM_test as DM_multi

from scipy import stats

warnings.filterwarnings("ignore")

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['lines.linewidth'] = 0.5

dist_list = ['normal','t','skewt','ged']
vol_list = ['GARCH', 'TARCH','GJR-GARCH','EGARCH','HARCH','FIGARCH','APARCH']
vol = vol_list
dist = dist_list

def main():
    print('#################',stock,'#################')
    getInput(stock)
    pvalue = [0.05]
    forecasts_year = "2021"
    result(pvalue,forecasts_year)
    ExpectedShortfall()
    assumptionCheck()
    makeResultdf()
    top_3_models = modelsRanked()
    top3graph(stock, df_VaR, df_results, returns, pvalue, vol_list, dist_list, forecasts_year, top_3_models)
    print('##################################')
          
def getInput(stock):
    global data
    global returns
    if stock.endswith('_data'):
        # Read the CSV file
        data = pd.read_csv(f'cryptos\{stock}.csv')
        # Convert the timestamp to a datetime object
        data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')
        # Set the 'Date' column as the index
        data = data.set_index('Date')
        # Calculate the daily percentage change in price
        data['returns'] = data['price'].pct_change() * 100
        data = data.dropna()
        # Drop the 'timestamp' column
        data = data.drop('timestamp', axis=1)
        returns = data['returns']
    else:
        yf.pdr_override()
        tickers = stock
        data = pdr.get_data_yahoo(tickers, start="1900-01-01", end="2023-04-06")
        data = data.tail(2521)
        returns = 100 * data['Adj Close'].pct_change().dropna()
    returns.name = "Returns"

def result(pvalue,forecasts_year):
    
    rets_2018 = returns[forecasts_year+"-04-06":].copy()
    rets_2018.name = "Returns"
    
    global df_VaR 
    df_VaR = pd.DataFrame(rets_2018)

    data = [0]*1
    global df_violation_ratio
    df_violation_ratio = pd.DataFrame(data)
    
    global df_aic
    df_aic = pd.DataFrame(data)
    
    global df_bic
    df_bic = pd.DataFrame(data)
    
    global df_VaRmean
    df_VaRmean = pd.DataFrame(data)
    
#stationarity    
    global df_adf
    df_adf = pd.DataFrame(data)
    global df_adf_p
    df_adf_p = pd.DataFrame(data)
    
    global df_pp
    df_pp = pd.DataFrame(data)
    global df_pp_p
    df_pp_p = pd.DataFrame(data)
#Independence
    global df_acf
    df_acf = pd.DataFrame(data)
    global df_acf_p
    df_acf_p = pd.DataFrame(data)
    
    global df_pacf
    df_pacf = pd.DataFrame(data)
    global df_pacf_p
    df_pacf_p = pd.DataFrame(data)
    
    global df_DW #autocorr
    df_DW = pd.DataFrame(data)
    
    global df_LB #autocorr
    df_LB = pd.DataFrame(data)
    global df_LB_p #autocorr
    df_LB_p = pd.DataFrame(data)
    
#normality
    global df_JB
    df_JB = pd.DataFrame(data)
    global df_JB_p
    df_JB_p = pd.DataFrame(data)
    
#comparison between models    
    global df_Loglik
    df_Loglik = pd.DataFrame(data)
    
    global df_Rolling
    df_Rolling = pd.DataFrame(data)
    
    global df_Scaling
    df_Scaling = pd.DataFrame(data)
    
    global df_mae
    df_mae = pd.DataFrame(data)
    
    global df_rmse
    df_rmse = pd.DataFrame(data)
    
    global df_assumptions
    df_assumptions = pd.DataFrame(data)
    
    global df_assumptions2
    df_assumptions2 = pd.DataFrame(data)
    
    rolling_start_date = '2021-04-06'  # Define the start date for the rolling window analysis
    rolling_end_date = "2023-04-06"
    rolling_start_date = pd.to_datetime(rolling_start_date)
    rolling_end_date = pd.to_datetime(rolling_end_date)
    forecasted_vol = []
    rolling_dates = df_VaR.loc[rolling_start_date:rolling_end_date].index

    
    for j in range(len(vol_list)):
        for i in range(len(dist_list)):
            model, result, q = Garch(vol_list[j],dist_list[i],pvalue)

            AIC_BIC(vol_list[j],dist_list[i],result)
            value_at_risk = -VaR(model,result,q,dist_list[i],forecasts_year)
            v_ratio = violation_ratio(value_at_risk,pvalue,rets_2018)
            df_violation_ratio[vol_list[j]+' '+dist_list[i]] = v_ratio
            df_VaR[vol_list[j]+' '+dist_list[i]] = value_at_risk
            df_VaRmean[vol_list[j]+' '+dist_list[i]] = df_VaR[vol_list[j]+' '+dist_list[i]].mean()
            #'''
            df_Scaling[vol_list[j]+' '+dist_list[i]] = np.mean(result.loglikelihood)
            forecast_volatility = result.conditional_volatility
            errors = np.abs(returns - forecast_volatility)
            mae = np.mean(errors)
            df_mae[vol_list[j]+' '+dist_list[i]] = mae
            rmse = np.sqrt(np.mean(errors ** 2))
            df_rmse[vol_list[j]+' '+dist_list[i]] = rmse
        
            #rolling
            forecasted_vol = Parallel(n_jobs=-1)(delayed(process_window)(t, model) for t in rolling_dates)
            realized_vol = df_VaR.loc[rolling_start_date:rolling_end_date, "Returns"].abs()
            rolling_mae = np.mean(np.abs(realized_vol - forecasted_vol))
            rolling_rmse = np.sqrt(np.mean((realized_vol - forecasted_vol) ** 2))
            df_Rolling[vol_list[j] + ' ' + dist_list[i]] = rolling_mae*0.5 + rolling_rmse*0.5
    df_Rolling = df_Rolling.transpose().drop(index = 0) 
    df_Scaling = df_Scaling.transpose().drop(index = 0)
    df_mae = df_mae.transpose().drop(index = 0)
    df_rmse = df_rmse.transpose().drop(index = 0)      
    #'''
    df_aic = df_aic.transpose().drop(index = 0)
    df_bic = df_bic.transpose().drop(index = 0)
    df_VaRmean = df_VaRmean.transpose().drop(index = 0) 
       
# Parallelize the computation (assuming you have access to multiple cores)
from joblib import Parallel, delayed

def process_window(t, model):
    window_size = 365  #252 is daily for stockmarket assets (incl etfs), 365 is for cryptos. Define the window size, can be 20 for monthly
    train_data = df_VaR.loc[:t].tail(window_size)
    train_returns = train_data["Returns"]

    # Fit the model and forecast 1-day-ahead volatility
    model_fitted = model.fit(disp="off", last_obs=train_returns.index[-1], options={'maxiter': 100})
    forecasts = model_fitted.forecast(horizon=1)
    return np.sqrt(forecasts.variance.dropna().values[-1, 0])            
         
def Garch(vol,dist,pvalue, model_params=None, power=None):
    # Specify the univariate GARCH model
    if vol == 'GJR-GARCH':
        model_params = {'p': 1, 'o': 1, 'q': 1}
    elif vol == 'TARCH':
        model_params = {'p': 1, 'o': 1, 'q': 1, 'power': 1.0}
    if vol == 'GJR-GARCH' or vol == 'TARCH':
        model = arch_model(returns, mean='Constant', vol='GARCH', dist=dist, **model_params)
    else:  # For standard GARCH
        model = arch_model(returns, mean='Constant', vol=vol, dist=dist)
    # Fit the model to the data
    result = model.fit(disp='off',last_obs="2021-01-01")
    resids = result.resid[:'2020']
    
    #assumption metrics
    ACF = acf(resids,nlags=5)
    df_acf[dist+vol] = ACF[0]
    df_acf_p[dist+vol] = ACF[1]
    PACF = acf(resids,nlags=5)
    df_pacf[dist+vol] = PACF[0]
    df_pacf_p[dist+vol] = PACF[1]    
    #autocorr / nonlinearity
    df_DW[dist+vol] = durbin_watson(resids)
    df_LB[dist+vol] = sm.stats.acorr_ljungbox(resids, lags=[1], return_df=False)[0]
    df_LB_p[dist+vol] = sm.stats.acorr_ljungbox(resids, lags=[1], return_df=False)[1]
    #goodness of fit
    df_Loglik[dist+vol] = result.loglikelihood
    
    if(dist == 'normal'):
        q = model.distribution.ppf(pvalue)
        #jarque_bera_test = stats.jarque_bera(st_resid)
        #print(jarque_bera_test)
        return model,result,q
    elif(dist == 't'):
        q = model.distribution.ppf(pvalue,result.params[-1:])
        return model,result,q
    elif (dist == 'skewt'):
        q = model.distribution.ppf(pvalue,result.params[-2:]) 
        return model,result,q
    elif(dist == 'ged'):
        q = model.distribution.ppf(pvalue,result.params[-1:])
        return model,result,q

def VaR(model,result,q,dist,forecasts_year):
    forecasts = result.forecast(start=forecasts_year+"-04-06", reindex=False) 
    cond_mean = forecasts.mean[forecasts_year+"-04-06":]
    
    cond_var = forecasts.variance[forecasts_year+"-04-06":]
    #forecast = result.forecast(horizon=1, method='simulation', simulations=1000)
    value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * q[None, :]
    return value_at_risk

def AIC_BIC(vol,dist,result):
    df_aic[vol+' '+dist] = result.aic
    df_bic[vol+' '+dist] = result.bic

def ExpectedShortfall():
    data = [999]*1
    global df_ES
    df_ES = pd.DataFrame(data)
    for z in range(len(vol_list)):
        for x in range(len(dist_list)):
            ES = 0
            counter = 0
            for i in range(len(df_VaR)):
                if df_VaR[vol_list[z]+' '+dist_list[x]].iloc[i] > df_VaR['Returns'].iloc[i]:
                    counter += 1
                    ES+= -(df_VaR['Returns'].iloc[i] - df_VaR[vol_list[z]+' '+dist_list[x]].iloc[i])
                    #print(ES)
            if counter == 0:
                ES = 99999
            else:
                ES = ES/counter
            df_ES[vol_list[z]+' ' + dist_list[x]] = ES
            
    #minimun_ES = df_ES.min(axis = 1,numeric_only = True)
    #model_ES = df_ES.idxmin(axis = 1)
    df_ES = df_ES.transpose().drop(index = 0)
    
    
    #print(minimun_ES,model_ES)
def violation_ratio(value_at_risk,pvalue,rets_2018):
    violation = 0
    no_violation = 0
    
    for i in range(len(value_at_risk)):
        if rets_2018[i] > value_at_risk[i]:
            no_violation +=1
        elif rets_2018[i] < value_at_risk[i]:
            violation +=1
         
    pvalue = pvalue[0]
    theoretical_v5 = pvalue*len(rets_2018)
    violation_ratio = violation/theoretical_v5
    return violation_ratio

def assumptionCheck():
    for j in range(len(dist_list)):
        for i in range(len(vol_list)):
            y_pred = df_VaR[vol_list[i]+' '+dist_list[j]] 
            #LB = acorr_ljungbox(y_pred, lags =[1])
            #df_LB[dist_list[j]+vol_list[i]] = LB 
            #stationarity
            ADF = adfuller(y_pred)
            df_adf[dist_list[j]+vol_list[i]] = ADF[0]
            df_adf_p[dist_list[j]+vol_list[i]] = ADF[1]
            PP = PhillipsPerron(y_pred)
            df_pp[dist_list[j]+vol_list[i]] = PP.stat
            df_pp_p[dist_list[j]+vol_list[i]] = PP.pvalue
            #normality
            JB = jarque_bera(y_pred)
            df_JB[dist_list[j]+vol_list[i]] = JB[0]
            df_JB_p[dist_list[j]+vol_list[i]] = JB[1]    

def makeResultdf():
    df_vr = df_violation_ratio.transpose().drop(index = 0)
    
    #print('Goodness of fit and Backtesting metrics')
    global df_results
    d= [0]*len(df_aic)
    df_results = pd.DataFrame(d)
    df_results.index = df_aic.index
    df_results['AIC']= df_aic
    df_results['BIC']= df_bic
    df_results['ES'] = df_ES
    df_results['violation Ratio'] = df_vr
    df_results['Mean Of VaR'] = df_VaRmean
    df_results['Scaling'] = df_Scaling
    df_results['Rolling'] = df_Rolling
    df_results['MAE'] = df_mae
    df_results['RMSE'] = df_rmse
    df_results = df_results.drop(columns = 0)
    #print(df_results.to_latex())
    
    print('assumption check')
    global df_assumptions
    d = [0]*len(df_adf.transpose())
    df_assumptions = pd.DataFrame(d)
    df_assumptions.index = df_adf.transpose().index
    df_assumptions['adf'] = df_adf.transpose()
    df_assumptions['adf p'] = df_adf_p.transpose()
    df_assumptions['PP'] = df_pp.transpose()
    df_assumptions['PP p'] = df_pp_p.transpose()
    df_assumptions = df_assumptions.drop(columns=0)
    df_assumptions = df_assumptions.drop(index=0)
    df_assumptions = df_assumptions.round(2)
    #print(df_assumptions.to_latex())
    
    print('assumption check2')
    global df_assumptions2
    d = [0]*len(df_DW.transpose())
    df_assumptions2 = pd.DataFrame(d)
    df_assumptions2.index = df_DW.transpose().index
    df_assumptions2['DW'] = df_DW.transpose()
    df_assumptions2['JB'] = df_JB.transpose()
    df_assumptions2['JB p'] = df_JB_p.transpose()
    df_assumptions2['LB'] = df_LB.transpose()
    df_assumptions2['LB p'] = df_LB_p.transpose()
    df_assumptions2 = df_assumptions2.drop(columns=0)
    df_assumptions2 = df_assumptions2.drop(index=0)
    df_assumptions2 = df_assumptions2.round(2)
    #print(df_assumptions2.to_latex())
    
    # Merging df_assumptions and df_assumptions2
    df_combined = pd.merge(df_assumptions, df_assumptions2, left_index=True, right_index=True)

    print(' Assumptions Check')
    print(df_combined.to_latex())
        
def modelsRanked():
    global df_metrics
    df_metrics = df_results

    # Normalize the 'Scaling' metric in the original dataframe
    df_metrics['Scaling'] = -df_metrics['Scaling']**-1
    df_metrics['Scaling'] = (df_metrics['Scaling'] - df_metrics['Scaling'].min()) / (df_metrics['Scaling'].max() - df_metrics['Scaling'].min())
    df_metrics.rename(columns={'Rolling': 'Roll'}, inplace=True)
    #print(df_metrics.round(3).to_latex())
    # Define columns to be selected for the combined metric
    selected_columns = ['AIC', 'BIC', 'Scaling', 'Roll', 'MAE', 'RMSE']

    # Create a new dataframe with selected metrics
    df_selected = df_metrics[selected_columns]

    # Normalizing the selected dataframe
    df_selected_normalized = (df_selected - df_selected.min()) / (df_selected.max() - df_selected.min())

    # Define the weights
    weights = {'AIC': 0.1, 'BIC': 0.1,  'Scaling': 0.1, 'Roll': 0.1, 'MAE':0.1, 'RMSE':0.1}

    # Apply the weights to calculate the 'C' metric
    df_selected_normalized['C'] = df_selected_normalized.apply(lambda row: sum(row[metric] * weights[metric] for metric in selected_columns), axis=1)

    # Add the C column to the original df_metrics DataFrame
    df_metrics['C'] = df_selected_normalized['C']
    
    # Rearrange columns to have C as the first column
    cols = df_metrics.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_metrics = df_metrics[cols]
    
    print(df_metrics.round(3).to_latex())
    
    # Drop unwanted columns
    df_metrics.drop(['ES', 'violation Ratio', 'Mean Of VaR'], axis=1, inplace=True)

    # Sort the dataframe based on the 'C' column and select the top 10 rows
    df_top10_original = df_metrics.sort_values('C', ascending=True).head(10)


    df_top10_original=df_top10_original.round(3)

    print(df_top10_original.to_latex())


    # Get the names of the top 3 models
    top_3_models = df_top10_original.index[:3].tolist()

    print(top_3_models)
    return top_3_models

def top3graph(stock, df_VaR, df_results, returns, pvalue, vol_list, dist_list, forecasts_year, top_3_models):
    # Extract the returns and the forecasts of the top 3 models
    data = df_VaR
    forecasts_top3 = data[top_3_models]
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(returns.index, returns.values, color='blue', label='Returns', linewidth=0.2)
    plt.title(f'Returns vs VaR of the top three models for {stock}')
    colors = ['green', 'red', 'purple']
    for i, (model_name, data) in enumerate(forecasts_top3.iteritems()):
        plt.plot(data.index, data.values, color=colors[i], label=model_name, linewidth=1)
    #plt.ylim(-600, 400)
    plt.xlim(pd.Timestamp('2021-04'), pd.Timestamp('2023-04'))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('VaR')
    # Save the plot
    plt.savefig(fr'C:\Users\guusb\OneDrive\Bureaublad\Research\{stock}.png')
    plt.show()
    
    # Parameter estimation for top 3 models
    rets_2018 = returns[forecasts_year+"-04-06":].copy()
    rets_2018.name = "Returns"
    for model_name in top_3_models:
        vol, dist = model_name.split()
        model, result, q = Garch(vol, dist, pvalue)
        print(f"Model: {model_name}, Vol: {vol}, Dist: {dist}, Params: {result.params}")
    
    #KS test
    for model_name in top_3_models:
        vol, dist = model_name.split()
        model, result, q = Garch(vol, dist, pvalue)
        check = model.fit()
        params = [check.params[-2], check.params[-1]]
        resids = check.resid
    
        try:
            cdf_values = model.distribution.cdf(resids,params)
            #Kolmogorov-Smirnov-test
            ks_statistic, p_value = stats.kstest(cdf_values, 'uniform')
            print('{:.3f}'.format(ks_statistic), '& {:.2f}'.format(p_value)) #\\\\ for cryptos
        
        except ValueError:
            pass  # Ignore the error and move to the next model_name

if __name__=="__main__":
    # List of stocks : 
    #ETFs = ['EUO','^DJI'.'VO','QQQ','DBE','UBS','BNO','UTSL','ERX', 'BOIL']
    #stocks = ['JNJ'.'MCD','MDLZ','MSFT','AMZN','BABA','TSLA','TUP','GME','FCEL']
    #later use and change some stuff in code before 'Bitcoin_data', 'Litecoin_data', 'Monero_data', 'XRP_data', 'Namecoin_data', 'Dogecoin_data', 'Feathercoin_data', 'BlackCoin_data', 'Gridcoin_data', 'Primecoin_data'
    list_stocks = ['Bitcoin_data', 'Litecoin_data', 'Monero_data', 'XRP_data', 'Namecoin_data', 'Dogecoin_data', 'Feathercoin_data', 'BlackCoin_data', 'Gridcoin_data', 'Primecoin_data']

    # Save the reference to the original standard output
    original_stdout = sys.stdout 

    for stock in list_stocks:
        # Modify the output path to include the current stock
        output_path = fr'C:\Users\guusb\OneDrive\Bureaublad\Research\console_output_{stock}.txt'

        # Open your desired file for writing
        with open(output_path, 'w') as file:
            sys.stdout = file  # Redirect the standard output to the file
            main()  # Call your main function
            sys.stdout = original_stdout  # Reset the standard output back to its original value
        
        print(f"Output for {stock} saved to {output_path}")