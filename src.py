import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def arima(data_frame: pd.DataFrame) -> None:
    data_frame['Date'] = pd.to_datetime(data_frame['Order Date'], format='%m/%d/%y %H:%M')
    data_frame['Order ID'] = pd.to_numeric(data_frame['Order ID'])
    data_frame['Quantity Ordered'] = pd.to_numeric(data_frame['Quantity Ordered'])
    data_frame['Price Each'] = pd.to_numeric(data_frame['Price Each'])
    data_frame.set_index('Date', inplace=True)


    # # Step 3: Visualize data
    # plt.figure(figsize=(12, 6))
    # plt.plot(data_frame)
    # plt.xlabel('Date')
    # plt.ylabel('Sales')
    # plt.title('Product Sales Time Series')
    # plt.show()

    # Step 4: Determine ARIMA orders (p, d, q)

    # # Step 5: Fit ARIMA model
    # model = sm.tsa.ARIMA(data, order=(p, d, q))
    # results = model.fit()

    # # Step 6: Evaluate model
    # forecast = results.forecast(steps=n_periods)
    # mae = np.mean(np.abs(data[-n_periods:] - forecast))
    # mse = np.mean((data[-n_periods:] - forecast) ** 2)
    # rmse = np.sqrt(mse)

    # # Step 7: Forecast sales

    # # Step 8: Visualize forecast
    # plt.figure(figsize=(12, 6))
    # plt.plot(data, label='Actual Sales')
    # plt.plot(forecast, label='Forecast')
    # plt.xlabel('Date')
    # plt.ylabel('Sales')
    # plt.title('Product Sales Forecast')
    # plt.legend()
    # plt.show()



def data_import(folder_path: str) -> pd.DataFrame:
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    final_df = None

    if len(csv_files) == 0:
        print("Files not found.")
    else:

        df_list = []
        for arquivo in csv_files:
            file_path = os.path.join(folder_path, arquivo)
            df = pd.read_csv(file_path, sep=',')
            df_cleaned = df.dropna()
            df_cleaned = df_cleaned[df_cleaned['Order Date'] != 'Order Date']
            df_list.append(df_cleaned)

        final_df = pd.concat(df_list, ignore_index=False).drop(columns=['Unnamed: 0'])

    final_df.to_csv('data/output/sales.csv')
    return final_df


def main() -> None:
    path = 'data/sales'
    df = data_import(path)
    breakpoint()
    arima(df)

main()