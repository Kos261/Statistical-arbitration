
def declining_installments(principal, annual_rate, n_periods, periods_per_year=1):
    """
    principal        – loan amount
    annual_rate      – annual interest rate, e.g. 0.07 for 7%
    n_periods        – number of payments
    periods_per_year – how many payments per year
    """
    r = annual_rate / periods_per_year          # interest per period
    principal_part = principal / n_periods      # constant principal each period

    schedule = []
    balance = principal

    for k in range(1, n_periods + 1):
        interest = balance * r                  # interest for current balance
        payment = principal_part + interest     # total payment
        balance -= principal_part               # new balance

        schedule.append({
            "payment_no": k,
            "principal_part": round(principal_part, 2),
            "interest_part": round(interest, 2),
            "payment": round(payment, 2),
            "remaining_balance": round(max(balance, 0), 2),
        })

    return schedule

def constant_installments(principal, annual_rate, n_periods, periods_per_year=1):
    """
    principal        – loan amount
    annual_rate      – annual interest rate, e.g. 0.07 for 7%
    n_periods        – number of payments
    periods_per_year – how many payments per year

    Annuity loan using formula:
    I = N * r / [ k * (1 - (k/(k+r))**n ) ]
    """
    r = annual_rate
    k = periods_per_year
    n = n_periods

    denom = k * (1 - (k / (k + r)) ** n)
    payment = principal * r / denom

    balance = principal
    schedule = []

    for k in range(1, n_periods + 1):
        interest = balance * r  # interest for current balance
        principal_part = payment - interest
        balance -= principal_part

        schedule.append({
            "payment_no": k,
            # "principal_part": round(principal_part, 2),
            # "interest_part": round(interest, 2),
            "payment": round(payment, 2),
            # "remaining_balance": round(max(balance, 0), 2),
        })

    return schedule




s1 = declining_installments(335_000, 0.07, 1)
print("Declining installments: 1 rata, 7%")
print(f"Rata{s1[0]["payment_no"]}:\t {s1[0]["payment"]}")


s2 = declining_installments(335_000, 0.035, 2)
print("Declining installments: 2 raty, 3.5%")
for rata in s2:
    print(f"Rata{rata["payment_no"]}:\t {rata["payment"]}")

s3 = constant_installments(335_000, 0.07, 1)
print("\n\nConstant installments: 1 rata, 7%")
print(f"Rata{s3[0]["payment_no"]}:\t {s3[0]["payment"]}")

s4 = constant_installments(335_000, 0.035, 2)
print("Constant installments: 2 raty, 3.5%")
for rata in s4:
    print(f"Rata{rata["payment_no"]}:\t {rata["payment"]}")