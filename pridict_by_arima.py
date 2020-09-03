# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import pearsonr
import math
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

#读数据
def read_data():
    #记录原始标号顺序
    index_list = []
    for j in range(0, 24):
        for i in range(1, 1321):
            index_list.append(i)

    # 读取汽车销量数据并排序
    sale_data_list = []
    sale_csv_data = pd.read_csv('train_sales_data.csv')
    sale_csv_data['index'] = pd.DataFrame(index_list)
    sale_csv_data.sort_values(by=['province', 'model', 'regYear', 'regMonth'], inplace=True)
    sale_csv_data = np.array(sale_csv_data)
    sale_csv_data = sale_csv_data.tolist()

    # 建立省份、车型编码、车身类型字典
    province_dict = dict()
    model_dict = dict()
    for row in sale_csv_data:
        if row[0] not in province_dict.keys():
            province_dict[row[0]] = len(province_dict.keys())
        if row[2] not in model_dict.keys():
            model_dict[row[2]] = len(model_dict.keys())
    print(len(province_dict),len(model_dict))

    # 建立分省份、车型sale数据树型list存储表
    for i in range(len(province_dict.keys())):
        tem_list = []
        sale_data_list.append(tem_list)
        for j in range(len(model_dict.keys())):
            tem_llist = []
            sale_data_list[i].append(tem_llist)
    for row in sale_csv_data:
        sale_data_list[province_dict[row[0]]][model_dict[row[2]]].append(row)

    # 打印
    # with open('sale_province_model.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for each_province in sale_data_list:
    #         for each_model in each_province:
    #             for row in each_model:
    #                 writer.writerow(row)
    return sale_data_list

#处理异常点
def solve_outliers(sale_data_list):
    for i in range(len(sale_data_list)):
        for j in range(len(sale_data_list[i])):
            sale_num = np.array([row[6] for row in sale_data_list[i][j]])
            flag = sale_num[12:] / sale_num[:12]
            Q1 = np.percentile(flag, 25)
            Q3 = np.percentile(flag, 75)

            #取2.5倍分位距外的数据为异常点
            max_num = Q3 + 2.5 * (Q3 - Q1)
            min_num = Q1 - 2.5 * (Q3 - Q1)
            for index in range(len(flag)):
                if flag[index] > max_num or flag[index] < min_num:
                    l_index = index - 1
                    r_index = index + 1
                    if l_index == -1:
                        num16 = sale_num[1]
                        num17 = sale_num[13]
                    elif r_index == 12:
                        num16 = sale_num[10]
                        num17 = sale_num[22]
                    else:
                        num16 = (sale_num[l_index] + sale_num[r_index]) / 2.0
                        num17 = (sale_num[12 + l_index] + sale_num[12 + r_index]) / 2.0

                    #异常点修正为两年份同月销量的几何平均数
                    if abs(sale_num[index] - num16) > abs(sale_num[12 + index] - num17):
                        sale_data_list[i][j][index][6] = (sale_data_list[i][j][12 + index][6] *
                                                          sale_data_list[i][j][index][6]) ** 0.5
                    else:
                        sale_data_list[i][j][12 + index][6] = (sale_data_list[i][j][12 + index][6] *
                                                               sale_data_list[i][j][index][6]) ** 0.5
    return sale_data_list

# 时间序列分解
def decompose(timeseries):
    # 返回包含三个部分trend（趋势部分），seasonal（季节性部分）和residual(残留部分)
    decomposition = seasonal_decompose(timeseries, model="multiplicative")
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # 作图
    # plt.subplot(411)
    # plt.plot(timeseries, label='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.plot(trend, label='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.plot(seasonal, label='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
    return trend, seasonal, residual

#求两list几何平均数
def list_square(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]*b[i]**0.5)
    return c

def pridect_data(sale_data_list):
    res_list = []
    for i in range(len(sale_data_list)):
        for j in range(len(sale_data_list[i])):
            # x = [row[5] for row in search_data_list[i][j]]
            y = [row[6] for row in sale_data_list[i][j]]
            year = [row[4] for row in sale_data_list[i][j]]
            month = [row[5] for row in sale_data_list[i][j]]
            TrainData = {'sale': y}
            TrainData = pd.DataFrame(TrainData)
            dates = {'year': year, 'month': month}
            dates = pd.DataFrame(dates)
            dates = dates[['year', 'month']].astype(int).astype(str)
            dataindex = dates["year"] + "-" + dates["month"]
            dataindex = dates_from_str(dataindex)
            TrainData = TrainData['sale']
            TrainData.index = pd.DatetimeIndex(dataindex)
            # 时间序列乘法分解
            trend, seasonal, residual = decompose(TrainData)
            # 除去季节因素
            TrainData = TrainData / seasonal
            # 对数化
            TrainData_log = np.log(TrainData)
            seasonal = np.array(list_square(seasonal[:4], seasonal[12:16]))
            # 移动平均
            # TrainData_log_rol_mean = TrainData_log.rolling(window=4).mean()
            # TrainData_log_rol_mean.dropna(inplace=True)
            # 差分
            # TrainData_log_rol_mean_diff_1 = TrainData_log.diff(1)
            # TrainData_log_rol_mean_diff_1.dropna(inplace=True)
            # TrainData_log_rol_mean_diff_2 = TrainData_log_rol_mean_diff_1.diff(1)
            # TrainData_log_rol_mean_diff_2.dropna(inplace=True)

            model = auto_arima(TrainData_log, error_action='ignore')
            #滚动迭代模型
            def forecast_one_step():
                fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
                return (
                    fc.tolist()[0],
                    np.asarray(conf_int).tolist()[0])
            forecasts = []
            confidence_intervals = []
            for new_ob in TrainData_log:
                fc, conf = forecast_one_step()
                forecasts.append(fc)
                confidence_intervals.append(conf)
                model.update(new_ob)
            # res_num = mean_squared_error(TrainData_log, forecasts)
            # # print(f"Mean squared error: {res_num}")
            # # print(f"SMAPE: {smape(TrainData_log, forecasts)}")
            # if res_num > 0.5 :
            #     forecast = list(y[12:16])
            # else :
            #     forecast = np.exp(model.predict(n_periods=4)) * seasonal[:4]

            #预测结果加入季节因素
            forecast = np.exp(model.predict(n_periods=4)) * seasonal[:4]
            print(forecast)
            res_list.append((sale_data_list[i][j][0][7], forecast))
    return res_list

if __name__ == "__main__":
    #读数据
    sale_data_list = read_data()
    #异常处理
    sale_data_list = solve_outliers(sale_data_list)
    #建立模型预测数据
    res_list = pridect_data(sale_data_list)
    #恢复原数据顺序
    res_list.sort()
    res_by_arima_list = []
    for i in range(0, 4):
        for row in res_list:
            res_by_arima_list.append(int(row[1][i]))
    predict_res = pd.DataFrame(res_by_arima_list, columns=['forecastVolum'])
    predict_res.to_csv('predict.csv')
    with open('res_by_arima.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in res_by_arima_list:
            writer.writerow(i)