from shop.pricing import calc_total


class Cart:
    def __init__(self):
        self.items = []

    def add(self, price):
        self.items.append(price)

    def total(self, tax=0.0):
        return calc_total(self.items, tax)
