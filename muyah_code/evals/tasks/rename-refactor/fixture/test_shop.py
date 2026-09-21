from shop.cart import Cart
from shop.pricing import calc_total


def test_total():
    assert calc_total([1, 2], 0.5) == 4.5


def test_cart():
    c = Cart()
    c.add(10)
    c.add(5)
    assert c.total() == 15
