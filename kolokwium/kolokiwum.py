# Mając w pliku kolokwium-dane-0912.csv dane szeregu czasowego opisującego
# pewien proces dokonaj prognozowania jego przebiegu na kolejny rok i zaproponuj
# funkcję oceniającą jokość tej prognozy. Jaki proces mogą opisywać te dane?
# Rozwiązanie w postaci notatnika Jupyter lub programu Python umieść w swoim
# repozytorium GitLab lub wyślij w DM. Kod powinien być dobrze opisany komentarzami i
# wykonywać się bez błędów w wybranym środowisku.


#Autor Konstanty Kłosiewicz 458597 MIMUW
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PoissonProcessDecomposer:
    # To decomposer z labu 8

    def __init__(self, df):
        self.df = df.copy()
        self.date_col = "Date"
        self.value_col = "count"

        # Przygotuj dane
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date").reset_index(drop=True)

        # Dodaj pomocnicze kolumny
        self.df['day_of_week'] = self.df["Date"].dt.dayofweek
        self.df['day_of_year'] = self.df["Date"].dt.dayofyear
        self.df['week_of_year'] = self.df["Date"].dt.isocalendar().week
        self.df['month'] = self.df["Date"].dt.month
        self.df['day'] = self.df["Date"].dt.day
        self.df['time_index'] = np.arange(len(self.df))

        # Wyniki dekompozycji
        self.components = {}
        self.lambda_estimated = None
        self.trend_model = None
        self.week_multipliers = np.ones(7)
        self.year_coeffs = {}
        self.holiday_multipliers = {}

    def fit(self):
        """
        1. Trend
        2. Sezonowość tygodniowa
        3. Sezonowość roczna
        4. Święta (outliers)
        5. Obliczenie λ(t) = trend(t) × week_pattern(t) × year_pattern(t) × holiday_effect(t) × noise(t)
        """
        self.estimate_trend()
        self.estimate_seasonality_week()
        self.estimate_seasonality_year()
        self.estimate_outliers()
        self.calculate_lambda()
        return self

    def validate_fit(self):
        actual = self.df["count"].values
        predicted = self.lambda_estimated(self.df['time_index'].values)

        # Metryki
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # R^2
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print("\nMETRICS:")
        print(f"MAE (Mean Absolute Error):        {mae:.2f}")
        print(f"RMSE (Root Mean Squared Error):   {rmse:.2f}")
        print(f"MAPE (Mean Abs Percentage Error): {mape:.2f}%")
        print(f"R² (Coefficient of Determination): {r2:.4f}")

        # # Wizualizacja dopasowania
        # plt.figure(figsize=(12, 6))
        # plt.plot(self.df['Date'], actual, label='Dane rzeczywiste', alpha=0.6)
        # plt.plot(self.df['Date'], predicted, label='Model (Fit)', color='red', alpha=0.8)
        # plt.title('Dopasowanie modelu dekompozycji')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.show()

    def estimate_trend(self):
        # Obliczanie trendu - regresja liniowa
        x = self.df['time_index']
        y = self.df["count"]
        coeffs = np.polyfit(x, y, 1)
        self.trend_model = np.poly1d(coeffs)
        self.df['trend_comp'] = self.trend_model(x)
        print(f"Trend: f(x) = ax + b\na = {coeffs[0]}, b = {coeffs[1]}")

    def estimate_seasonality_week(self):
        # Usuwam trend (ponieważ model trend(t) × week_pattern(t)... to wystarczy dzielić)
        self.df['detrended'] = self.df['count'] / self.df['trend_comp']
        week_grp = self.df.groupby('day_of_week')['detrended'].mean()
        # Normalizacja
        week_grp = week_grp / week_grp.mean()
        self.week_multipliers = week_grp.values

        self.df['week_comp'] = self.df['day_of_week'].map(lambda x: self.week_multipliers[x])
        print("Sezonowość tygodniowa (mnożniki Pon-Niedz):")
        print(np.round(self.week_multipliers, 3))

    def estimate_seasonality_year(self):
        # Usuwamy trend i tydzień
        self.df['deweeked'] = self.df['detrended'] / self.df['week_comp']
        # Szereg Fouriera : 1 + sum( a_n * sin + b_n * cos )
        t = self.df['day_of_year'].values
        y_resid = self.df['deweeked'].values
        order = 3
        # Budujemy macierz cech dla regresji liniowej: [sin_1, cos_1, sin_2, cos_2...]
        X_fourier = []
        for i in range(1, order + 1):
            X_fourier.append(np.sin(2 * np.pi * i * t / 365.25))
            X_fourier.append(np.cos(2 * np.pi * i * t / 365.25))

        X_matrix = np.column_stack(X_fourier)
        # Fitujemy odchylenia od 1 (bo to mnożniki)
        # y_resid ~ 1 + Fourier
        # więc fitujemy Fourier ~ (y_resid - 1)
        target = y_resid - 1
        coeffs, _, _, _ = np.linalg.lstsq(X_matrix, target, rcond=None)

        self.year_coeffs = coeffs

        # Obliczamy komponent roczny dla ramki
        y_fourier = np.dot(X_matrix, coeffs)
        self.df['year_comp'] = 1 + y_fourier

        # Zabezpieczenie, żeby mnożnik nie był ujemny
        self.df['year_comp'] = self.df['year_comp'].clip(lower=0.1)

    def estimate_outliers(self):
        # Usuwamy trend, seasonality
        predicted_so_far = (self.df['trend_comp'] * self.df['week_comp'] * self.df['year_comp'])
        residuals = self.df['count'] / predicted_so_far

        std_resid = residuals.std()
        mean_resid = residuals.mean()

        upper_limit = mean_resid + 3 * std_resid
        lower_limit = mean_resid - 3 * std_resid

        outliers = self.df[(residuals > upper_limit) | (residuals < lower_limit)].copy()

        for _, row in outliers.iterrows():
            key = (row['Date'].month, row['Date'].day)
            # Mnożnik to zaobserwowana wartość / oczekiwana wartość
            multiplier = row['count'] / (row['trend_comp'] * row['week_comp'] * row['year_comp'])
            self.holiday_multipliers[key] = multiplier

        print(f"Wykryto {len(self.holiday_multipliers)} dni specjalnych (outlierów).")

    def calculate_lambda(self):
        """
        Tworzy funkcję, która przyjmuje tablicę indeksów czasowych (lub int)
        i zwraca prognozę. Musimy mieć możliwość mapowania index -> data.
        """
        start_date = self.df['Date'].min()

        def predict_func(time_indices):
            # Konwersja indeksów na daty
            dates = start_date + pd.to_timedelta(time_indices, unit='D')
            # Trend
            trend = self.trend_model(time_indices)
            # Week
            day_of_weeks = dates.dayofweek
            week_pattern = np.array([self.week_multipliers[d] for d in day_of_weeks])
            # Year
            day_of_years = dates.dayofyear
            order = len(self.year_coeffs) // 2
            y_fourier = np.zeros(len(dates))
            for i in range(1, order + 1):
                idx = (i - 1) * 2
                y_fourier += self.year_coeffs[idx] * np.sin(2 * np.pi * i * day_of_years / 365.25)
                y_fourier += self.year_coeffs[idx + 1] * np.cos(2 * np.pi * i * day_of_years / 365.25)
            year_pattern = 1 + y_fourier
            year_pattern = np.clip(year_pattern, 0.1, None)
            # Holidays
            holiday_pattern = np.ones(len(dates))
            for i, d in enumerate(dates):
                key = (d.month, d.day)
                if key in self.holiday_multipliers:
                    holiday_pattern[i] = self.holiday_multipliers[key]

            # Wynik końcowy
            return trend * week_pattern * year_pattern * holiday_pattern

        self.lambda_estimated = predict_func

if __name__ == '__main__':
    df = pd.read_csv('kolokwium-dane-0912.csv', sep=';')
    df['count'] = df['count'].values
    df['Date'] = pd.to_datetime(df['Date'])
    print(df.head(5))

    poiss = PoissonProcessDecomposer(df)
    poiss.fit()
    poiss.validate_fit()

    future_days = 365
    last_t = poiss.df['time_index'].max()
    future_t = np.arange(last_t + 1, last_t + 1 + future_days)

    forecast = poiss.lambda_estimated(future_t)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['count'], label='Historia')
    plt.plot(pd.date_range('2021-01-01', periods=365), forecast, label='Prognoza', color='green')
    plt.title("Prognoza na rok 2022")
    plt.legend()
    plt.show()