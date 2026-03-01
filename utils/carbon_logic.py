import pandas as pd

# CO2e factors (kg CO2 per kg of waste)
# Sources: EPA / World Bank data averages
IMPACT_DATA = {
    "Biodegradable": {"loss": 0.9, "gain": 0.05, "desc": "Compostable - returns nutrients to soil."},
    "Recyclable": {"loss": 2.5, "gain": 0.4, "desc": "Can be repurposed into new raw materials."},
    "Hazardous": {"loss": 5.0, "gain": 1.2, "desc": "Requires specialized chemical treatment."},
    "Landfill": {"loss": 3.0, "gain": 2.8, "desc": "Non-recyclable; generates methane in pits."}
}

def get_carbon_metrics(label):
    data = IMPACT_DATA.get(label, {"loss": 1.0, "gain": 0.5, "desc": "General waste category."})
    return {
        "co2_landfill": data["loss"],
        "co2_recycled": data["gain"],
        "co2_saved": round(data["loss"] - data["gain"], 2),
        "info": data["desc"]
    }
