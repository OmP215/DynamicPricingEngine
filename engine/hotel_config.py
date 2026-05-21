import json
from engine import PricingParams

def load_hotel_parameters():
    with open('hotel_pricing_parameters.json', 'r') as f:
        data = json.load(f)

    # clean up metadata
    for key in ['estimated_at', 'peak_demand']:
        data.pop(key, None)

    return PricingParams(**data)

