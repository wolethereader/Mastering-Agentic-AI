#!/usr/bin/env python3
from mcp.server import FastMCP

NUTRITION_DB = {
    "apple":          {"calories": 95,  "protein_g": 0.5,  "carbs_g": 25, "fat_g": 0.3, "fibre_g": 4.4},
    "chicken breast": {"calories": 165, "protein_g": 31.0, "carbs_g": 0,  "fat_g": 3.6, "fibre_g": 0},
    "brown rice":     {"calories": 216, "protein_g": 5.0,  "carbs_g": 45, "fat_g": 1.8, "fibre_g": 3.5},
    "broccoli":       {"calories": 55,  "protein_g": 3.7,  "carbs_g": 11, "fat_g": 0.6, "fibre_g": 5.1},
    "salmon":         {"calories": 208, "protein_g": 28.0, "carbs_g": 0,  "fat_g": 10,  "fibre_g": 0},
    "oats":           {"calories": 150, "protein_g": 5.0,  "carbs_g": 27, "fat_g": 2.5, "fibre_g": 4.0},
}

server = FastMCP('diet-coach-nutrition', instructions='Nutrition lookup tools for the AI Diet Coach.')

@server.tool(name='lookup_nutrition', description='Return macro-nutrient data for a food item per 100 g serving.', structured_output=True)
def lookup_nutrition(food_item: str) -> dict:
    key = food_item.strip().lower()
    data = NUTRITION_DB.get(key)
    if data:
        return {'food': key, **data}
    matches = [k for k in NUTRITION_DB if key in k or k in key]
    if matches:
        match = matches[0]
        return {'food': match, 'note': f"closest match for '{food_item}'", **NUTRITION_DB[match]}
    return {'error': f"'{food_item}' not found"}

if __name__ == '__main__':
    server.run(transport='stdio')
