def get_nested_value(data, target_key):
    """
    Recursively search for the target_key in a nested dictionary and return its value.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value
            result = get_nested_value(value, target_key)
            if result is not None:
                return result
    return None