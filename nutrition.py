
def get_nutrition_info(food_name):
    """
    Returns density (g/cm3) and calories_per_100g (kcal) for a given food class.
    Data provided by user.
    """
    # Normalize name
    name = food_name.lower().strip()
    
    # Database based on user input
    # Format: 'name': {'density': float, 'calories_per_100g': float}
    nutrition_db = {
        'egg': {'density': 1.03, 'calories_per_100g': 143},
        'rice': {'density': 0.85, 'calories_per_100g': 130}, 
        'qingcai': {'density': 0.975, 'calories_per_100g': 17.5}, 
        'qiezi': {'density': 0.95, 'calories_per_100g': 25},
        'tofu': {'density': 1.00, 'calories_per_100g': 87}, 
        'tudou': {'density': 1.05, 'calories_per_100g': 77},
        'tudousi': {'density': 0.75, 'calories_per_100g': 77}, # Adjusted for bulk density (air gaps)
        'xihongshi': {'density': 0.85, 'calories_per_100g': 18}, # Adjusted for chunks/voids
        'lajiao': {'density': 0.95, 'calories_per_100g': 25}, 
        'jirou': {'density': 1.05, 'calories_per_100g': 165},
        'xilanhua': {'density': 0.95, 'calories_per_100g': 34}
    }
    
    # Default fallback
    default_info = {'density': 1.0, 'calories_per_100g': 100}
    
    return nutrition_db.get(name, default_info)
