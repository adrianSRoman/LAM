import math 
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def az_ele_from_source(ref_point, src_point):
    """
    Calculates the azimuth and elevation between a reference point and a source point in 3D space.
    Args:
    ref_point (list): A list of three floats representing the x, y, and z coordinates of the reference point.
    src_point (list): A list of three floats representing the x, y, and z coordinates of the other point.
    Returns:
    A tuple of two floats representing the azimuth and elevation angles in degrees plus the distance between the reference and source points.
    """
    dx = src_point[0] - ref_point[0]
    dy = src_point[1] - ref_point[1]
    dz = src_point[2] - ref_point[2]

    azimuth = math.degrees(math.atan2(dy, dx))
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    elevation = math.degrees(math.asin(dz / distance))
    return azimuth, elevation, distance


def get_mic_xyz():
  """
  Get em32 microphone coordinates in 3D space
  """
  return [(3 - 3) * 0.5, (3 - 3) * 0.5, (2 - 2) * 0.3 + 1.5]


def plot_points_in_room(source_coords, labels):
    # Extract x, y, z coordinates
    x_coords = [coord[0] for coord in source_coords]
    y_coords = [coord[1] for coord in source_coords]
    z_coords = [coord[2] for coord in source_coords]

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    scatter = ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

    # Set labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Set title
    ax.set_title('3D Plot of Points in Room')

    # Create annotation object
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = labels[ind["ind"][0]]
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()


# Reference METU outter trayectory:  bottom outter trayectory
REF_OUT_TRAJ = ["034", "024", "014", "004", "104", "204",
                        "304", "404", "504", "604", "614", "624",
                        "634", "644", "654", "664", "564", "464",
                        "364", "264", "164", "064", "054", "044"]


top_height = 5
source_coords, rirs = [], []
sources_xyz = []
mic_xyz = get_mic_xyz()

labels = [f"{num[0]}{num[1]}{str(4-(int(num[2])-height))}" for height in range(top_height) for num in REF_OUT_TRAJ]

rir_id = 0
# Outter trayectory: bottom to top
for height in range(0, top_height):
    for num in REF_OUT_TRAJ:
        # Coords computed based on documentation.pdf from METU Sparg
        x = (3 - int(num[0])) * 0.5
        y = (3 - int(num[1])) * 0.5
        z = (2 - (int(num[2])-height)) * 0.3 + 1.5
        source_xyz = [x, y, z] # note -1 since METU is flipped up-side-down
        sources_xyz.append((x, y, z))
        azim, elev, _ = az_ele_from_source(mic_xyz, source_xyz)
        elev *= -1 # Account for elevation being swapped in METU
        source_coords.append((rir_id, azim, elev))

plot_points_in_room(sources_xyz, labels)
