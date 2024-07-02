"""
Generates the mesh based on the voxels.
Added Arguments:
material dict: dicitionary mapping material to object
class_names: pulls in the new class names created after object splitting
class_colors, materials, materila_swapped -> arguments to get material names and their colours
Edited this file such that the material name is retrived based on the object assigned to each voxel.
Sets the vertices the colour based on the material
The face is set the colour of the most frequent material within that area.
"""


import math
import numpy as np
import openmesh as om
from statistics import mean
from skimage import measure
from collections import Counter
from scipy.spatial import KDTree
from collections import Counter

# class_colors = [
#     (0.1, 0.1, 0.1, 1),
#     (0.0649613, 0.467197, 0.0667303, 1),
#     (0.1, 0.847035, 0.1, 1),
#     (0.0644802, 0.646941, 0.774265, 1),
#     (0.131518, 0.273524, 0.548847, 1),
#     (1, 0.813553, 0.0392201, 1),
#     (1, 0.490452, 0.0624932, 1),
#     (0.657877, 0.0505005, 1, 1),
#     (0.0363214, 0.0959549, 0.548847, 1),
#     (0.316852, 0.548847, 0.186899, 1),
#     (0.548847, 0.143381, 0.0045568, 1),
#     (1, 0.241096, 0.718126, 1),
#     (0.9, 0.0, 0.0, 1),
#     (0.4, 0.0, 0.0, 1),
#     (0.3, 0.3, 0.3, 1)
#     ]

# class_colors_ordered = [
#     (0.0644802, 0.646941, 0.774265, 1), #wall
#     (0.1, 0.847035, 0.1, 1), #floor
#     (0.0649613, 0.467197, 0.0667303, 1), #ceiling
#     (0.548847, 0.143381, 0.0045568, 1), #furniture
#     (0.316852, 0.548847, 0.186899, 1), #table
#     (0.657877, 0.0505005, 1, 1), #sofa
#     (1, 0.490452, 0.0624932, 1),  # bed
#     (0.0363214, 0.0959549, 0.548847, 1), #tvs
#     (1, 0.813553, 0.0392201, 1), #chair
#     (1, 0.241096, 0.718126, 1), #objects
#     (0.131518, 0.273524, 0.548847, 1), #window
#     (0.1, 0.1, 0.1, 1), #empty
#     (0.9, 0.0, 0.0, 1), #error1
#     (0.4, 0.0, 0.0, 1), #error2
#     (0.3, 0.3, 0.3, 1) #error3
#     ]

class_colors = [
    (19/255, 17/255, 17/255, 1), # asphalt
    (202/255, 198/255, 144/255, 1), # ceramic
    (186/255, 200/255, 238/255, 1), # concrete
    (0/255, 0/255, 200/255, 1), # fabric
    (89/255, 125/255, 49/255, 1), # foliage
    (0/255, 70/255, 0/255, 1), # food
    (187/255, 129/255, 156/255, 1), # glass
    (208/255, 206/255, 72/255, 1), # metal
    (98/255, 39/255, 69/255, 1), # paper
    (102/255, 102/255, 102/255, 1), # plaster
    (76/255, 74/255, 95/255, 1), # plastic
    (16/255, 16/255, 68/255, 1), # rubber
    (68/255, 65/255, 38/255, 1), # soil
    (117/255, 214/255, 70/255, 1), # stone
    (221/255, 67/255, 72/255, 1), # water
    (92/255, 133/255, 119/255, 1) # wood
]

# class_names = ["empty", "ceiling", "floor", "wall", "window", "chair", "bed", "sofa",
#                "table", "tvs", "furniture", "objects", "error1", "error2", "error3"]
material_labels = {
    0: "asphalt", 1: "ceramic", 2: "concrete", 3: "fabric", 4: "foliage",
    5: "food", 6: "glass", 7: "metal", 8: "paper", 9: "plaster", 10: "plastic",
    11: "rubber", 12: "soil", 13: "stone", 14: "water", 15: "wood"
}

material_labels_swapped = {
    "asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
    "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
    "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15
}

def draw_mesh(name, vert_array, cat_array, voxel_list, v_unit, material_dict, class_names):
    print("Exporting mesh to "+name)

    vu = v_unit * 4

    # Use marching cubes to obtain the surface mesh
    verts, faces, normals, values = measure.marching_cubes(vert_array, spacing=(1.0, 1.0, 1.0))

    mesh = om.TriMesh()

    mesh.request_face_colors()

    vert_arr = np.array([])
    face_arr = np.array([])
    color_arr = np.array([])
    color_counter_arr = np.array([])

    tree = KDTree(voxel_list, balanced_tree=True)

    print("           ")
    print("Adding mesh vertices...")
    for x in range(len(verts)):
        vertex = verts[x]
        vertex = (vertex - 500) * vu
        vh = mesh.add_vertex(vertex)
        vert_arr = np.append(vert_arr, vh)
        query = tree.query(vertex, k=1)
        try:
            mtl = cat_array[query[1]]
            corresponding_material = material_dict.get(class_names[mtl])
            clr = material_labels_swapped.get(corresponding_material)
            color_arr = np.append(color_arr, cat_array[query[1]])
            mesh.set_color(vh, class_colors[clr])
        except:
            print(query, mtl)
        # color_arr = np.append(color_arr, cat_array[query[1]])
        # mesh.set_color(vh, class_colors[cat_array[query[1]]])
        print("%d%%" % (100 * x / len(verts)), end="\r")

    print("Adding mesh faces...")
    for y in range(len(faces)):
        face = faces[y]
        fh = mesh.add_face([vert_arr[face[0]], vert_arr[face[1]], vert_arr[face[2]]])
        color_counter = Counter([color_arr[face[0]], color_arr[face[1]], color_arr[face[2]]])
        color = color_counter.most_common(1)[0][0]
        corresponding_material = material_dict.get(class_names[int(color)])
        clr = material_labels_swapped.get(corresponding_material)
        try:
            mesh.set_color(fh, class_colors[int(clr)])
        except:
            print()
        # if len(color_counter) == 1:
        #     color = color_counter.most_common(1)[0][0]
        #     mesh.set_color(fh, class_colors[int(color)])
        # else:
        #     for idx in range(len(class_colors_ordered)):
        #         if class_colors_ordered[idx] in [class_colors[int(index)] for index in set(color_counter)]:
        #             mesh.set_color(fh, class_colors_ordered[idx])
        #             break
        print("%d%%" % (100 * y / len(faces)), end="\r")
    print("           ")
    
    om.write_mesh(str(name), mesh, face_color=True)

    '''
    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 1000)

    plt.tight_layout()
    plt.show(block=True)
    '''