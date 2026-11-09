"""Order pricing: bulk discounts and sales tax."""

from decimal import ROUND_HALF_UP, Decimal

# (minimum quantity, discount rate), checked from the largest tier down.
BULK_TIERS: list[tuple[int, Decimal]] = [
    (100, Decimal("0.10")),
    (50, Decimal("0.05")),
]
TAX_RATE = Decimal("0.08")
CENT = Decimal("0.01")


def bulk_discount_rate(quantity: int) -> Decimal:
    for minimum, rate in BULK_TIERS:
        if quantity >= minimum:
            return rate
    return Decimal("0")


def line_total(unit_price: float, quantity: int) -> Decimal:
    """Price of one order line after the bulk discount, before tax."""
    gross = Decimal(str(unit_price)) * quantity
    discounted = gross * (1 - bulk_discount_rate(quantity))
    return discounted.quantize(CENT, rounding=ROUND_HALF_UP)


def order_total(lines: list[tuple[float, int]]) -> Decimal:
    """Total of (unit_price, quantity) lines including tax, rounded to cents."""
    subtotal = sum((line_total(price, qty) for price, qty in lines), Decimal("0"))
    tax = (subtotal * TAX_RATE).quantize(CENT, rounding=ROUND_HALF_UP)
    return subtotal + tax
