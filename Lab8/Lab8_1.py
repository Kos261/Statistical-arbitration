# Lab I.81 Ile można zarobić?
# W dołączonym do zadania pliku \Verb+data-i.81.csv+ znajdują się notowania
# kursu Bitcoina podane w USD (czyli BTC/USD).  Jaka była maksymalna stopa zwrotu z
# inwestycji w okresie od północy 11 grudnia do godziny 23:59 15 grudnia 2023 roku?
# Pomiń koszty transakcyjne.

import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("data-i.81.csv")
    df["time"] = pd.to_datetime(df["time"], format='ISO8601')
    df["rate"] = pd.to_numeric(df["rate"])

    date1 = pd.to_datetime("2023-12-11 00:00:00").tz_localize('UTC')
    date2 = pd.to_datetime("2023-12-15 23:59:59.999999").tz_localize('UTC')

    # Poprawne filtrowanie z operatorami logicznymi
    df_filtered = df[(df["time"] >= date1) & (df["time"] <= date2)]
    print(df_filtered.head(5))

    min_rate = df_filtered["rate"].min()
    max_rate = df_filtered["rate"].max()

    stopa_zwrotu = (max_rate - min_rate) / min_rate * 100

    print("Maksymalna stopa zwrotu: ", stopa_zwrotu, "%")