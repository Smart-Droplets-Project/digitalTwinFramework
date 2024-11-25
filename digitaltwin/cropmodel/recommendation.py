import datetime


def fill_it_up(crop_model_output: dict):
    year = crop_model_output["day"].year
    fertilization_dates = [datetime.date(year, 4, 1), datetime.date(year, 5, 1)]
    result = 0
    if crop_model_output["day"] in fertilization_dates:
        navail = crop_model_output["NAVAIL"]
        result = max(0, 80 - navail)
    return result
