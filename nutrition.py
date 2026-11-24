"""
Stage 6: Cooked-food density library and USDA SR28 energy coefficients.
Matches Tables 5 and 6 of the NutriQuant manuscript across the 10 categories.
"""

NUTRITION_DB = {
    "rice": {
        "display_name": "Rice",
        "density": 0.73,
        "density_sd": 0.04,
        "calories_per_100g": 130.0,
        "energy_kcal_per_g": 1.300,
        "usda_ndb_no": "20045",
        "usda_ref": "White long-grain rice, enriched, cooked",
    },
    "tudou": {
        "display_name": "Potato",
        "density": 1.03,
        "density_sd": 0.03,
        "calories_per_100g": 86.0,
        "energy_kcal_per_g": 0.860,
        "usda_ndb_no": "11367",
        "usda_ref": "Potato flesh, boiled without skin or salt",
    },
    "tudousi": {
        "display_name": "Shredded potato",
        "density": 0.99,
        "density_sd": 0.03,
        "calories_per_100g": 86.0,
        "energy_kcal_per_g": 0.860,
        "usda_ndb_no": "11367",
        "usda_ref": "Potato flesh, boiled without skin or salt",
    },
    "lajiao": {
        "display_name": "Pepper",
        "density": 0.89,
        "density_sd": 0.03,
        "calories_per_100g": 28.0,
        "energy_kcal_per_g": 0.280,
        "usda_ndb_no": "11334",
        "usda_ref": "Green sweet pepper, boiled, drained, no salt",
    },
    "qingcai": {
        "display_name": "Leafy vegetables",
        "density": 0.87,
        "density_sd": 0.05,
        "calories_per_100g": 12.0,
        "energy_kcal_per_g": 0.120,
        "usda_ndb_no": "11117",
        "usda_ref": "Pak-choi, boiled, drained, no salt",
    },
    "xilanhua": {
        "display_name": "Broccoli",
        "density": 0.92,
        "density_sd": 0.03,
        "calories_per_100g": 35.0,
        "energy_kcal_per_g": 0.350,
        "usda_ndb_no": "11091",
        "usda_ref": "Broccoli, boiled, drained, no salt",
    },
    "tofu": {
        "display_name": "Tofu",
        "density": 1.03,
        "density_sd": 0.02,
        "calories_per_100g": 76.0,
        "energy_kcal_per_g": 0.760,
        "usda_ndb_no": "16427",
        "usda_ref": "Regular tofu with calcium sulfate, raw",
    },
    "qiezi": {
        "display_name": "Eggplant",
        "density": 0.94,
        "density_sd": 0.04,
        "calories_per_100g": 35.0,
        "energy_kcal_per_g": 0.350,
        "usda_ndb_no": "11210",
        "usda_ref": "Eggplant, boiled, drained, no salt",
    },
    "jirou": {
        "display_name": "Chicken",
        "density": 1.10,
        "density_sd": 0.03,
        "calories_per_100g": 165.0,
        "energy_kcal_per_g": 1.650,
        "usda_ndb_no": "05064",
        "usda_ref": "Chicken breast meat, roasted, no skin",
    },
    "egg": {
        "display_name": "Egg",
        "density": 1.01,
        "density_sd": 0.02,
        "calories_per_100g": 155.0,
        "energy_kcal_per_g": 1.550,
        "usda_ndb_no": "01129",
        "usda_ref": "Whole egg, hard-boiled",
    },
}

DEFAULT_NUTRITION = {
    "display_name": "Unknown",
    "density": 1.00,
    "density_sd": 0.05,
    "calories_per_100g": 100.0,
    "energy_kcal_per_g": 1.000,
    "usda_ndb_no": "N/A",
    "usda_ref": "Generic cooked food fallback",
}


def get_nutrition_info(food_name):
    """
    Returns cooked-food density (g/cm^3), USDA SR28 energy (kcal/100g & kcal/g),
    and reference metadata for a given NutriQuant-DB food class.
    """
    name = str(food_name).lower().strip()
    return dict(NUTRITION_DB.get(name, DEFAULT_NUTRITION))
