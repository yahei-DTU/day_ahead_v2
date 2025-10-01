# Creating mesh/grid points for DK2 and DK1 and displaying them as tables.
import numpy as np
import pandas as pd

# Define bounding boxes (lat_min, lon_min, lat_max, lon_max)
# DK2 (East Denmark: Zealand, Lolland-Falster, Bornholm area)
dk2_bbox = (54.9, 11.5, 56.2, 15.2)

# DK1 (West Denmark: Jutland and Funen)
dk1_bbox = (54.55, 7.0, 57.7, 11.5)

# Choose resolution: 3 rows (lat) x 4 cols (lon) = 12 sample points each
n_lat = 3
n_lon = 4

def mesh_points(bbox, n_lat, n_lon):
    lat_min, lon_min, lat_max, lon_max = bbox
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    pts = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            pts.append({
                "zone": None,
                "row": i+1,
                "col": j+1,
                "latitude": round(float(lat), 6),
                "longitude": round(float(lon), 6)
            })
    return pts

dk2_points = mesh_points(dk2_bbox, n_lat, n_lon)
for p in dk2_points:
    p["zone"] = "DK2"

dk1_points = mesh_points(dk1_bbox, n_lat, n_lon)
for p in dk1_points:
    p["zone"] = "DK1"

df_dk2 = pd.DataFrame(dk2_points)
df_dk1 = pd.DataFrame(dk1_points)

print("DK1 Mesh Points:")
print(df_dk1)

print("DK2 Mesh Points:")
print(df_dk2)

# Also print a small summary for convenience
df_summary = pd.concat([df_dk2, df_dk1], ignore_index=True)
df_summary.to_csv("dk_mesh_points.csv", index=False)
print("Generated 12 DK2 points and 12 DK1 points and saved to dk_mesh_points.csv")
