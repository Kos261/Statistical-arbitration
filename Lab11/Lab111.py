def check_arbitrage(start_amount, path, rates):
    """
    start_amount: kwota początkowa
    path: lista walut w kolejności, np. ['USD', 'GBP', 'EUR', 'USD']
    rates: słownik z kursami
    """
    current_amount = start_amount
    print(f"Start: {start_amount:.2f} {path[0]}")

    for i in range(len(path) - 1):
        curr_from = path[i]
        curr_to = path[i + 1]

        pair_direct = f"{curr_from}/{curr_to}"  # Np. USD/GBP (rzadko kwotowane w ten sposób)
        pair_inverse = f"{curr_to}/{curr_from}"  # Np. GBP/USD (standard)

        if pair_inverse in rates:
            rate = rates[pair_inverse]['sell']
            new_amount = current_amount / rate
            operation = f"/ {rate} (Ask)"

        elif pair_direct in rates:
            # Sytuacja: Mamy EUR, tabela podaje EUR/USD.
            # Chcemy sprzedać EUR (walutę bazową), żeby dostać USD (kwotowaną).
            # Bank od nas KUPUJE (Buy/Bid).
            # Działanie: Mnożymy przez Bid.
            rate = rates[pair_direct]['buy']
            new_amount = current_amount * rate
            operation = f"* {rate} (Bid)"

        else:
            print(f"Brak notowań dla pary {curr_from}-{curr_to}")
            return None

        print(f"  {curr_from} -> {curr_to}: {current_amount:.4f} {operation} = {new_amount:.4f} {curr_to}")
        current_amount = new_amount

    profit = current_amount - start_amount
    roi = (profit / start_amount) * 100

    print(f"Koniec: {current_amount:.2f} {path[-1]}")
    print(f"Wynik: {profit:+.2f} ({roi:+.2f}%)")
    print("-" * 30)
    return current_amount


# --- DANE Z TWOJEGO ZADANIA ---

# Słownik odwzorowujący tabelę z zadania.
# Klucze to pary w formacie BAZOWA/KWOTOWANA (np. 1 EUR = x USD)
market_rates = {
    # Rynek A
    'EUR/USD': {'buy': 1.0202, 'sell': 1.0284},
    'GBP/USD': {'buy': 1.5718, 'sell': 1.5844},
    # Rynek B
    'EUR/GBP': {'buy': 0.6324, 'sell': 0.6401},
    'USD/GBP': {'buy': 0.6299, 'sell': 0.6375}  # Tę parę też masz w tabeli B (1 USD = x GBP)
}

# --- URUCHOMIENIE ---

# Ścieżka 1: Zgodnie z ruchem wskazówek zegara (USD -> EUR -> GBP -> USD)
path1 = ['USD', 'EUR', 'GBP', 'USD']
check_arbitrage(1000, path1, market_rates)

# Ścieżka 2: Przeciwnie do ruchu wskazówek (USD -> GBP -> EUR -> USD)
# To jest ta zyskowna ścieżka, którą wyliczyliśmy
path2 = ['USD', 'GBP', 'EUR', 'USD']
check_arbitrage(1000, path2, market_rates)