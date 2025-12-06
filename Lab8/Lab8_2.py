# Krok 1. Wygenerować proces i wykonać wykres na bazie poniższych fragmentów kodu.
# Krok 2. Proszę znaleźć parametry trendu w wygenerowanych danych. Poniżej pomocniczy kod.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PoissonProcessDecomposer:
    """
    Klasa do dekompozycji szeregu czasowego na komponenty 
    niehomogenicznego procesu Poissona.
    
    Zakładamy model:
    λ(t) = trend(t) × week_pattern(t) × year_pattern(t) × holiday_effect(t) × noise(t)
    
    gdzie calls ~ Poisson(λ(t))
    """
    
    def __init__(self, df, date_col='date', value_col='calls'):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame z danymi
        date_col : str
            Nazwa kolumny z datami
        value_col : str
            Nazwa kolumny z liczbą połączeń
        """
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        
        # Przygotuj dane
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col).reset_index(drop=True)
        
        # Dodaj pomocnicze kolumny
        self.df['day_of_week'] = self.df[date_col].dt.dayofweek
        self.df['day_of_year'] = self.df[date_col].dt.dayofyear
        self.df['week_of_year'] = self.df[date_col].dt.isocalendar().week
        self.df['month'] = self.df[date_col].dt.month
        self.df['day'] = self.df[date_col].dt.day
        self.df['time_index'] = np.arange(len(self.df))
        
        # Wyniki dekompozycji
        self.components = {}
        self.lambda_estimated = None

    def fit(self, trend_method='linear', seasonality_method='fourier'):
        """
        1. Trend
        2. Sezonowość tygodniowa
        3. Sezonowość roczna
        4. Święta (outliers)
        5. Obliczenie λ(t)
        """

        self.estimate_trend(method=trend_method)
        return self

    def validate_fit(self):
        actual = self.df["calls"].values
        predicted = self.lambda_estimated(self.df['time_index'].values)

        # Metryki
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # R^2
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Sprawdź czy reszty są zgodne z Poissonem
        residuals = actual - predicted
        print("METRICS:")
        print(f"MAE (Mean Absolute Error):        {mae:.2f}")
        print(f"RMSE (Root Mean Squared Error):   {rmse:.2f}")
        print(f"MAPE (Mean Abs Percentage Error): {mape:.2f}%")
        print(f"R² (Coefficient of Determination): {r2:.4f}")


        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }


    def estimate_trend(self, method='linear'):
        x = self.df['time_index']
        y = self.df['calls']
        coeffs = np.polyfit(x, y, 1)
        self.lambda_estimated = np.poly1d(coeffs)
        print(f"Trend: f(x) = ax + b\na = {coeffs[0]}, b = {coeffs[1]}")



if __name__ == "__main__":
    # 1. Wygeneruj syntetyczne dane (symulacja prawdziwych danych)
    print("Generowanie syntetycznych danych...")
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    data = []
    
    for i, date in enumerate(dates):
        day_of_week = date.dayofweek
        day_of_year = date.dayofyear
        
        trend = 100 + i * 0.1
        week_pattern = 0.6 if day_of_week in [5, 6] else 1.0
        year_pattern = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        is_holiday = (date.month == 12 and date.day == 25) or \
                     (date.month == 1 and date.day == 1)
        holiday_effect = 0.3 if is_holiday else 1.0
        random_effect = np.random.uniform(0.9, 1.1)
        
        lambda_t = trend * week_pattern * year_pattern * holiday_effect * random_effect
        calls = np.random.poisson(lambda_t)
        
        data.append({'date': date, 'calls': calls})
    
    df = pd.DataFrame(data)
    print(f"✓ Wygenerowano {len(df)} dni danych\n")

    # plt.plot(df['date'], df['calls'])
    # plt.title("Calls per day")
    # plt.show()


    decom = PoissonProcessDecomposer(df)
    decom.fit()
    decom.validate_fit()