def calc_total(prices, tax=0.0):
    subtotal = sum(prices)
    return round(subtotal * (1 + tax), 2)
