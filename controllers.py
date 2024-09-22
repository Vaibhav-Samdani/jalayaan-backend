from flask import Flask, jsonify, current_app as app
from flask import request, render_template, redirect, session
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from heapq import heappush, heappop
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from heapq import heappush, heappop
from scipy.ndimage import gaussian_filter1d


def get_data(data):
    return jsonify(data)

def getML(start, dest) :
    if (start == "Mumbai" and dest == "Lakshdweep") or (start == "Lakshdweep" and dest == "Mumbai") :
            

            # Mumbai and Lakshadweep Islands coordinates
            mumbai_coords = (19.0760, 72.8777)  # Mumbai: Latitude, Longitude
            lakshadweep_coords = (10.5667, 72.6420)  # Lakshadweep Islands: Latitude, Longitude

            # # Define the extent for the map (focused on Mumbai and Lakshadweep region)
            # lat_bounds = [5, 25]  # Latitude bounds for visualization (adjusted to cover Lakshadweep)
            # lon_bounds = [70, 75]  # Longitude bounds for visualization (covers Mumbai and Lakshadweep region)
            # grid_size = (200, 200)  # Grid size for fine resolution

            # # Create a lat/lon grid
            # lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], grid_size[0])
            # lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], grid_size[1])

            # Updated extent for a larger area of the Indian Ocean
            lat_bounds = [-40, 30]  # Latitude bounds for visualization (extends further south)
            lon_bounds = [40, 100]  # Longitude bounds for visualization (extends further east)

            # Update the lat/lon grid accordingly
            grid_size = (100, 100)  # Keep the grid size constant for simplicity

            lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], grid_size[0])
            lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], grid_size[1])

            # Function to find the grid point for a given latitude and longitude
            def find_grid_point(lat, lon, lat_grid, lon_grid):
                lat_idx = np.argmin(np.abs(lat_grid - lat))
                lon_idx = np.argmin(np.abs(lon_grid - lon))
                return (lat_idx, lon_idx)

            start_point = find_grid_point(mumbai_coords[0], mumbai_coords[1], lat_grid, lon_grid)
            end_point = find_grid_point(lakshadweep_coords[0], lakshadweep_coords[1], lat_grid, lon_grid)

            # Generate a land mask (1 = land, 0 = sea, 1 for impassable regions due to weather)
            land_mask = np.zeros(grid_size)

            # Mark land on the map (rough approximation)
            land_mask[0:40, :] = 1  # land in the northern part (representing India's landmass)
            land_mask[:, 0:10] = 1  # land on the left (western coast)

            # Simulate impassable regions in the sea due to bad weather, waves, and wind
            land_mask[100:110, 50:60] = 1  # Example impassable area due to weather
            land_mask[30:40, 130:140] = 1  # Another impassable area

            # A* algorithm to find the shortest sea path
            def heuristic(a, b):
                return np.sqrt((a[0] - b[0])*2 + (a[1] - b[1])*2)

            # Modify neighbors to reduce diagonal movement influence
            def a_star(start, goal, land_mask):
                rows, cols = land_mask.shape
                open_list = []
                heappush(open_list, (0, start))
                came_from = {start: None}
                g_score = {start: 0}
                f_score = {start: heuristic(start, goal)}

                # Reduced diagonal movement influence (fewer diagonal neighbors)
                neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Only allow cardinal directions

                while open_list:
                    _, current = heappop(open_list)

                    if current == goal:
                        path = []
                        while current:
                            path.append(current)
                            current = came_from[current]
                        return path[::-1]

                    for d in neighbors:
                        neighbor = (current[0] + d[0], current[1] + d[1])

                        if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                            if land_mask[neighbor] == 1:  # Skip land and impassable regions
                                continue

                            tentative_g_score = g_score[current] + heuristic(current, neighbor)
                            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                                came_from[neighbor] = current
                                g_score[neighbor] = tentative_g_score
                                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                                heappush(open_list, (f_score[neighbor], neighbor))

                return None

            # Find the sea route using A*
            path = a_star(start_point, end_point, land_mask)

            # Check if a valid path was found
            if path is None:
                print("No valid sea route could be found!")
            else:
                # Convert path to lat/lon coordinates
                lat_route = [lat_grid[p[0]] for p in path]
                lon_route = [lon_grid[p[1]] for p in path]

                # Plot the route on a map
                fig = plt.figure(figsize=(10, 6))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())

                # Add coastlines and other geographical features
                ax.coastlines()
                ax.add_feature(cfeature.LAND, edgecolor='black')
                ax.add_feature(cfeature.BORDERS, linestyle=':')

                # Plot Mumbai and Lakshadweep on the map
                ax.plot(mumbai_coords[1], mumbai_coords[0], color='red', marker='o', markersize=8, label='Mumbai', transform=ccrs.PlateCarree())
                ax.plot(lakshadweep_coords[1], lakshadweep_coords[0], color='green', marker='o', markersize=8, label='Lakshadweep', transform=ccrs.PlateCarree())

                # Plot the sea route
                ax.plot(lon_route, lat_route, color='yellow', linewidth=2, label='Sea Route', transform=ccrs.PlateCarree())

                # Highlight impassable regions due to weather
                for i in range(grid_size[0]):
                    for j in range(grid_size[1]):
                        if land_mask[i, j] == 1:
                            ax.plot(lon_grid[j], lat_grid[i], color='blue', marker='x', markersize=4, transform=ccrs.PlateCarree())

                # Add labels and a title
                # plt.title("Smoother Sea Route from Mumbai to Lakshadweep Islands (Avoiding Land and Adverse Weather)")
                plt.legend(loc='upper right')
                path = "../team-jalayaan/src/images/mum_lak.png"
                plt.savefig(path)
                return path
                # plt.show()


def getMM(start,dest):
    # Mumbai and Maldives (Malé) coordinates
        mumbai_coords = (19.0760, 72.8777)  # Mumbai: Latitude, Longitude
        maldives_coords = (4.1755, 73.5093)  # Malé, Maldives: Latitude, Longitude

        # # Define the extent for the map (focused on Indian Ocean, including Mumbai and Maldives)
        # lat_bounds = [-5, 22]  # Latitude bounds for visualization (adjusted to include the Maldives)
        # lon_bounds = [65, 80]  # Longitude bounds for visualization (covers Mumbai and Maldives region)
        # grid_size = (100, 100)


        # Updated extent for a larger area of the Indian Ocean
        lat_bounds = [-40, 30]  # Latitude bounds for visualization (extends further south)
        lon_bounds = [40, 100]  # Longitude bounds for visualization (extends further east)

        # Update the lat/lon grid accordingly
        grid_size = (100, 100)  # Keep the grid size constant for simplicity

        lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], grid_size[0])
        lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], grid_size[1])

        # Reuse the rest of the code unchanged from here...

        # # Create a lat/lon grid
        # lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], grid_size[0])
        # lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], grid_size[1])

        # Function to find the grid point for a given latitude and longitude
        def find_grid_point(lat, lon, lat_grid, lon_grid):
            lat_idx = np.argmin(np.abs(lat_grid - lat))
            lon_idx = np.argmin(np.abs(lon_grid - lon))
            return (lat_idx, lon_idx)

        start_point = find_grid_point(mumbai_coords[0], mumbai_coords[1], lat_grid, lon_grid)
        end_point = find_grid_point(maldives_coords[0], maldives_coords[1], lat_grid, lon_grid)

        # Generate a land mask (1 = land, 0 = sea, 1 for impassable regions due to weather)
        land_mask = np.zeros(grid_size)

        # Mark land on the map (rough approximation)
        land_mask[0:30, :] = 1  # land in the northern part (representing India's landmass)
        land_mask[:, 0:10] = 1  # land on the left (western coast)
        land_mask[:, 90:100] = 1  # land on the right (eastern coast)

        # Simulate impassable regions in the sea due to bad weather, waves, and wind
        # You can adjust the areas where the weather is impassable

        # Larger, irregular impassable regions (representing harsh weather areas)
        land_mask[45:55, 35:55] = 1  # A large area due to waves
        land_mask[25:35, 65:75] = 1  # Another irregular area due to winds
        land_mask[65:70, 20:45] = 1  # A third area affected by weather

        # Add smaller, scattered impassable regions
        for i in range(10, 20, 2):
            for j in range(60, 70, 3):
                land_mask[i, j] = 1  # Random patches due to strong winds

        # A* algorithm to find the shortest sea path
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])*2 + (a[1] - b[1])*2)  # Fixed formula

        def a_star(start, goal, land_mask):
            rows, cols = land_mask.shape
            open_list = []
            heappush(open_list, (0, start))
            came_from = {start: None}
            g_score = {start: 0}
            f_score = {start: heuristic(start, goal)}

            while open_list:
                _, current = heappop(open_list)

                if current == goal:
                    path = []
                    while current:
                        path.append(current)
                        current = came_from[current]
                    return path[::-1]

                # Allow for diagonal movements
                neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                
                for d in neighbors:
                    neighbor = (current[0] + d[0], current[1] + d[1])

                    if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                        if land_mask[neighbor] == 1:  # Skip land and impassable regions
                            continue

                        tentative_g_score = g_score[current] + heuristic(current, neighbor)
                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                            heappush(open_list, (f_score[neighbor], neighbor))

            return None

        # Find the sea route using A*
        path = a_star(start_point, end_point, land_mask)

        # Check if a valid path was found
        if path is None:
            print("No valid sea route could be found!")
        else:
            # Convert path to lat/lon coordinates
            lat_route = [lat_grid[p[0]] for p in path]
            lon_route = [lon_grid[p[1]] for p in path]

            # Smooth the route to make it more curved
            lat_route_smooth = gaussian_filter1d(lat_route, sigma=2)
            lon_route_smooth = gaussian_filter1d(lon_route, sigma=2)

            # Plot the route on a map
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())

            # Add coastlines and other geographical features
            ax.coastlines()
            ax.add_feature(cfeature.LAND, edgecolor='black')  # Only land features
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            # Plot Mumbai and Maldives on the map
            ax.plot(mumbai_coords[1], mumbai_coords[0], color='red', marker='o', markersize=8, label='Mumbai', transform=ccrs.PlateCarree())
            ax.plot(maldives_coords[1], maldives_coords[0], color='green', marker='o', markersize=8, label='Maldives', transform=ccrs.PlateCarree())

            # Plot the smoothed sea route
            ax.plot(lon_route_smooth, lat_route_smooth, color='yellow', linewidth=2, label='Sea Route', transform=ccrs.PlateCarree())

            # Highlight impassable regions due to weather
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    if land_mask[i, j] == 1:
                        ax.plot(lon_grid[j], lat_grid[i], color='blue', marker='x', markersize=4, transform=ccrs.PlateCarree())

            # Set the background color of the entire figure to white
            fig.patch.set_facecolor('white')

            # Add labels and a title
            # plt.title("Smoothed Sea Route from Mumbai to Maldives (Avoiding Land and Adverse Weather)")
            plt.legend(loc='upper right')
            path = "../team-jalayaan/src/images/mum_lak.png"
            plt.savefig(path)
            return path;


def getLM(start,dest):
    # Lakshadweep (Agatti Island) and Maldives (Malé) coordinates
        lakshadweep_coords = (10.7865, 72.1890)  # Agatti Island, Lakshadweep: Latitude, Longitude
        maldives_coords = (4.1755, 73.5093)      # Malé, Maldives: Latitude, Longitude

        # Updated extent for a larger area of the Indian Ocean
        lat_bounds = [-40, 30]  # Latitude bounds for visualization (extends further south)
        lon_bounds = [40, 100]  # Longitude bounds for visualization (extends further east)

        # Update the lat/lon grid accordingly
        grid_size = (100, 100)  # Keep the grid size constant for simplicity

        lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], grid_size[0])
        lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], grid_size[1])

        # Function to find the grid point for a given latitude and longitude
        def find_grid_point(lat, lon, lat_grid, lon_grid):
            lat_idx = np.argmin(np.abs(lat_grid - lat))
            lon_idx = np.argmin(np.abs(lon_grid - lon))
            return (lat_idx, lon_idx)

        # Define start and end points
        start_point = find_grid_point(lakshadweep_coords[0], lakshadweep_coords[1], lat_grid, lon_grid)
        end_point = find_grid_point(maldives_coords[0], maldives_coords[1], lat_grid, lon_grid)

        # Generate a land mask (1 = land, 0 = sea, 1 for impassable regions due to weather)
        land_mask = np.zeros(grid_size)

        # Mark land for Lakshadweep and the Maldives
        land_mask[0:20, 60:100] = 1  # Land in Lakshadweep
        land_mask[0:20, 0:30] = 1    # Land in Maldives

        # Add impassable regions due to weather
        land_mask[30:50, 40:80] = 1  # Example impassable weather areas

        # A* algorithm to find the shortest sea path
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)  # Euclidean distance

        def a_star(start, goal, land_mask):
            rows, cols = land_mask.shape
            open_list = []
            heappush(open_list, (0, start))
            came_from = {start: None}
            g_score = {start: 0}
            f_score = {start: heuristic(start, goal)}

            while open_list:
                _, current = heappop(open_list)

                if current == goal:
                    path = []
                    while current:
                        path.append(current)
                        current = came_from[current]
                    return path[::-1]

                # Allow for diagonal movements
                neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                
                for d in neighbors:
                    neighbor = (current[0] + d[0], current[1] + d[1])

                    if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                        if land_mask[neighbor] == 1:  # Skip land and impassable regions
                            continue

                        tentative_g_score = g_score[current] + heuristic(current, neighbor)
                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                            heappush(open_list, (f_score[neighbor], neighbor))

            return None

        # Find the sea route using A*
        path = a_star(start_point, end_point, land_mask)

        # Check if a valid path was found
        if path is None:
            print("No valid sea route could be found!")
        else:
            # Convert path to lat/lon coordinates
            lat_route = [lat_grid[p[0]] for p in path]
            lon_route = [lon_grid[p[1]] for p in path]

            # Smooth the route to make it more curved
            lat_route_smooth = gaussian_filter1d(lat_route, sigma=2)
            lon_route_smooth = gaussian_filter1d(lon_route, sigma=2)

            # Plot the route on a map
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())

            # Add coastlines and other geographical features
            ax.coastlines()
            ax.add_feature(cfeature.LAND, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            # Plot Lakshadweep and Maldives on the map
            ax.plot(lakshadweep_coords[1], lakshadweep_coords[0], color='red', marker='o', markersize=8, label='Lakshadweep', transform=ccrs.PlateCarree())
            ax.plot(maldives_coords[1], maldives_coords[0], color='green', marker='o', markersize=8, label='Maldives', transform=ccrs.PlateCarree())

            # Plot the smoothed sea route
            ax.plot(lon_route_smooth, lat_route_smooth, color='yellow', linewidth=2, label='Sea Route', transform=ccrs.PlateCarree())

            # Highlight impassable regions due to weather
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    if land_mask[i, j] == 1:
                        ax.plot(lon_grid[j], lat_grid[i], color='blue', marker='x', markersize=4, transform=ccrs.PlateCarree())

            # Set the background color of the entire figure to white
            fig.patch.set_facecolor('white')

            # Add labels and a title
            # plt.title("Smoothed Sea Route from Lakshadweep to Maldives (Avoiding Land and Adverse Weather)")
            plt.legend(loc='upper right')

            # Save the plot
            path = "../team-jalayaan/src/images/mum_lak.png"
            plt.savefig(path)
            return path;


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return render_template("./index.html")
    elif request.method=="POST":
        # Get the JSON data from the frontend
        data = request.get_json()

    # Access specific fields from the JSON
        start = data.get('start')
        dest = data.get('dest')
        print(start,dest)

        if(start == "Mumbai" and dest == "Lakshdweep") or (start == "Lakshdweep" and dest == "Mumbai"):
            path = getML(start,dest)
            return jsonify({"path" : path,"found":"true"})
        elif(start=="Mumbai" and dest == "Maldives") or (start=="Maldives" and dest == "Mumbai"):
            path = getMM(start,dest)
            return jsonify({"path":path,"found":"true"})
        elif(start=="Lakshdweep" and dest == "Maldives") or (start=="Maldives" and dest == "Lakshdweep"):
            path = getLM(start,dest)
            return jsonify({"path":path,"found":"true"})

        
        