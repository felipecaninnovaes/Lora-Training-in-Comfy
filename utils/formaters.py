def value_formater(value: str) -> str:
    formatted_value = str(format(value, "e")).rstrip('0').rstrip()
    return ''.join(c for c in formatted_value if not (c == '0'))