import pandas as pd

# might need to reference as AI-generated
def convert_instance(path):
    # try: 
    #     df = pd.read_csv(path, sep=r"\s+", engine="python", header=None)
    # except FileNotFoundError:
    #     print(f"File '{path}' not found.")
    # except Exception as e:
    #     print(f"An error occured: {e}")
    
    with open(path, "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    num_nodes, episode_length, vehicle_capacity = lines[0]
    parameters = {
        "num_nodes": int(num_nodes),
        "episode_length": int(episode_length),
        "vehicle_capacity": float(vehicle_capacity)

    }
    supplier_info = lines[1]
    supplier = {
        "id": int(supplier_info[0]),
        "x_cord": float(supplier_info[1]),
        "y_cord": float(supplier_info[2]),
        "initial_inventory" : float(supplier_info[3]),
        "production_rate": float(supplier_info[4]),
        "holding_cost": float(supplier_info[5])
    }

    retailers_info = lines[2:]
    retailers = pd.DataFrame(retailers_info, columns=["id", "x_cord", "y_cord", "initial_inventory", "max_capacity", "min_capacity", "demand", "holding_cost"]).astype(float)
    retailers["id"] = retailers["id"].astype(int)

    return {"parameters": parameters, "supplier": supplier, "retailers": retailers}