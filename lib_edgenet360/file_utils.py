"""
Edits for material mapping:

Passes in material encoded voxel grid into obj_export
Implemented object splitting in obj_export
Using object assignement and material encoded voxel grid,
implemented majority voting to assign material to each object

"""


#from lib_csscnet.py_cuda import *
import numpy as np
from sklearn.utils import shuffle
from fnmatch import fnmatch
import os
from tensorflow.keras.utils import to_categorical
import threading
from collections import defaultdict, Counter
from skimage import measure
from lib_edgenet360.mesh_utils import draw_mesh


material_labels_swapped = {
    "asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
    "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
    "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15
}

material_labels = {
    0: "asphalt", 1: "ceramic", 2: "concrete", 3: "fabric", 4: "foliage",
    5: "food", 6: "glass", 7: "metal", 8: "paper", 9: "plaster", 10: "plastic",
    11: "rubber", 12: "soil", 13: "stone", 14: "water", 15: "wood"
}


color_plate = {
    0: [119, 17, 17], 1: [202, 198, 144], 2: [186, 200, 238], 3: [0, 0, 200], 4: [89, 125, 49],
    5: [16, 68, 16], 6: [187, 129, 156], 7: [208, 206, 72], 8: [98, 39, 69], 9: [102, 102, 102],
    10: [76, 74, 95], 11: [16, 16, 68], 12: [68, 65, 38], 13: [117, 214, 70], 14: [221, 67, 72],
    15: [92, 133, 119]
}

grayscale_plate = {0: 47, 1: 193, 2: 200, 3: 23, 4: 106, 5: 41, 6: 149, 7: 191, 8: 60, 9: 102,
                   10: 77, 11: 22, 12: 63, 13: 169, 14: 114, 15: 119}


def get_file_prefixes_from_path(data_path, criteria="*.bin"):
    prefixes = []

    for (path, subdirs, files) in os.walk(data_path):
        print(path)
        print(subdirs)
        for name in files:
            if fnmatch(name, criteria):
                prefixes.append(os.path.join(path, name)[:-4])

    prefixes.sort()

    return prefixes


class threadsafe_generator(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, gen):
        self.gen = gen
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.gen.__next__()


def threadsafe(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_generator(f(*a, **kw))
    return g


@threadsafe
def ftsdf_generator(file_prefixes, batch_size=4, shuff=False, shape=(240, 144, 240), down_scale = 4):  # write the definition of your data generator

    while True:
        x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
        rgb_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 3))
        y_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale, 12))
        w_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale))
        batch_count = 0
        if shuff:
            file_prefixes_s = shuffle(file_prefixes)
        else:
            file_prefixes_s = file_prefixes

        for count, file_prefix in enumerate(file_prefixes_s):


            vox_tsdf, vox_rgb, segmentation_label, vox_weights = process(file_prefix, voxel_shape=(240, 144, 240), down_scale=4)

            x_batch[batch_count] = vox_tsdf.reshape((shape[0], shape[1], shape[2],1))
            rgb_batch[batch_count] = vox_rgb.reshape((shape[0], shape[1], shape[2],3))
            y_batch[batch_count] = to_categorical(segmentation_label.reshape((60, 36, 60,1)), num_classes=12)
            w_batch[batch_count] = vox_weights.reshape((60, 36, 60))
            batch_count += 1
            if batch_count == batch_size:
                yield [x_batch, rgb_batch], y_batch, w_batch
                x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1)) #channels last
                rgb_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 3)) #channels last
                y_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale, 12))
                w_batch = np.zeros((batch_size, shape[0]//down_scale, shape[1]//down_scale, shape[2]//down_scale))
                batch_count = 0

        if(batch_count > 0):
            yield [x_batch[:batch_count], rgb_batch[:batch_count]], y_batch[:batch_count], w_batch[:batch_count]


@threadsafe
def preproc_generator(file_prefixes, batch_size=4, shuff=False, aug=False, vol=False, shape=(240, 144, 240), down_scale = 4, type="rgb"):  # write the definition of your data generator

    down_shape = (shape[0] // down_scale,  shape[1] // down_scale, shape[2] // down_scale)

    while True:
        x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
        if type == "rgb":
            rgb_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 3))
        elif type == "edges":
            edges_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
        y_batch = np.zeros((batch_size, down_shape[0], down_shape[1], down_shape[2], 12))
        if vol:
            vol_batch = np.zeros((batch_size, down_shape[0], down_shape[1], down_shape[2]))
        batch_count = 0
        if shuff:
            file_prefixes_s = shuffle(file_prefixes)
        else:
            file_prefixes_s = file_prefixes

        for count, file_prefix in enumerate(file_prefixes_s):

            npz_file = file_prefix + '.npz'
            loaded = np.load(npz_file)

            #print(file_prefix)

            vox_tsdf  = loaded['tsdf']
            if type == "rgb":
                vox_rgb = loaded['rgb']
            elif type == "edges":
                vox_edges = loaded['edges']
            vox_label  = loaded['lbl']
            vox_weights = loaded['weights']
            if vol:
                vox_vol = loaded['vol']

            x_batch[batch_count] = vox_tsdf
            if vol:
                vol_batch[batch_count] = vox_vol

            if aug:

                aug_v = np.random.normal(loc=1, scale=0.05, size=3)

            else:
                aug_v=np.array([1.,1.,1.])

            if type == "rgb":
                rgb_batch[batch_count] = np.clip(vox_rgb * aug_v, 0., 1.)
            elif type == "edges":
                edges_batch[batch_count] = vox_edges

            labels = to_categorical(vox_label, num_classes=12)
            weights =  np.repeat(vox_weights,12,axis=-1).reshape((down_shape[0], down_shape[1], down_shape[2], 12))
            y_batch[batch_count] = labels * (weights+1)
            batch_count += 1
            if batch_count == batch_size:
                if type == "rgb":
                    if vol:
                        yield [x_batch, rgb_batch], y_batch, vol_batch
                    else:
                        yield [x_batch, rgb_batch], y_batch
                elif type == "edges":
                    if vol:
                        yield [x_batch, edges_batch], y_batch, vol_batch
                    else:
                        yield [x_batch, edges_batch], y_batch
                elif type == "depth":
                    if vol:
                        yield x_batch, y_batch, vol_batch
                    else:
                        yield x_batch, y_batch

                x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1)) #channels last
                if type == "rgb":
                    rgb_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 3))
                elif type == "edges":
                    edges_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
                y_batch = np.zeros((batch_size, down_shape[0], down_shape[1], down_shape[2], 12))
                if vol:
                    vol_batch = np.zeros((batch_size, down_shape[0], down_shape[1], down_shape[2]))
                batch_count = 0

        if batch_count > 0:
            if type == "rgb":
                if vol:
                    yield [x_batch[:batch_count], rgb_batch[:batch_count]], y_batch[:batch_count],vol_batch[:batch_count]
                else:
                    yield [x_batch[:batch_count], rgb_batch[:batch_count]], y_batch[:batch_count]
            elif type == "edges":
                if vol:
                    yield [x_batch[:batch_count], edges_batch[:batch_count]], y_batch[:batch_count],vol_batch[:batch_count]
                else:
                    yield [x_batch[:batch_count], edges_batch[:batch_count]], y_batch[:batch_count]
            elif type == "depth":
                if vol:
                    yield x_batch[:batch_count], y_batch[:batch_count],vol_batch[:batch_count]
                else:
                    yield x_batch[:batch_count], y_batch[:batch_count]


@threadsafe
def evaluate_generator(file_prefixes, batch_size=4, shuff=False, shape=(240, 144, 240), down_scale = 4):  # write the definition of your data generator

    while True:
        x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
        y_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale, 12))
        f_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale),dtype=np.int32)
        batch_count = 0
        if shuff:
            file_prefixes_s = shuffle(file_prefixes)
        else:
            file_prefixes_s = file_prefixes

        for count, file_prefix in enumerate(file_prefixes_s):


            vox_tsdf, segmentation_label, vox_flags = process_evaluate(file_prefix, voxel_shape=(240, 144, 240), down_scale=4)

            x_batch[batch_count] = vox_tsdf.reshape((shape[0], shape[1], shape[2],1))
            y_batch[batch_count] = to_categorical(segmentation_label.reshape((60, 36, 60,1)), num_classes=12)
            f_batch[batch_count] = vox_flags.reshape((60, 36, 60))
            batch_count += 1
            if batch_count == batch_size:
                yield x_batch, y_batch, f_batch
                x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1)) #channels last
                y_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale, 12))
                f_batch = np.zeros((batch_size, shape[0]//down_scale, shape[1]//down_scale, shape[2]//down_scale),dtype=np.int32)
                batch_count = 0

        if(batch_count > 0):
            yield x_batch[:batch_count], y_batch[:batch_count],f_batch[:batch_count]

def voxel_export(name, vox, shape, tsdf_range=None, sample=None):

    from array import array
    import struct
    import random




    vox = vox.reshape(shape)

    coord_x=array('i')
    coord_y=array('i')
    coord_z=array('i')
    voxel_v=array('f')

    count = 0

    for x in range(shape[0]):
        for y in range (shape[1]):
             for z in range(shape[2]):
                 if ((tsdf_range is None) and (vox[z,y,x]!=0)) or \
                    ((not tsdf_range is None) and  ((vox[z,y,x]<=tsdf_range[0] or vox[z,y,x]>=tsdf_range[1]) and vox[z,y,x]!=1)):
                     if sample is None or sample > random.random():
                         coord_x.append(x)
                         coord_y.append(y)
                         coord_z.append(z)
                         voxel_v.append(vox[z,y,x])
                         count += 1
    f = open(name, 'wb')
    f.write(struct.pack("i", count))
    f.write(struct.pack(str(count)+"i", *coord_x))
    f.write(struct.pack(str(count)+"i", *coord_y))
    f.write(struct.pack(str(count)+"i", *coord_z))
    f.write(struct.pack(str(count)+"f", *voxel_v))
    f.close()

    return count


def rgb_voxel_export(name, vox, shape, sample=None):

    from array import array
    import struct
    import random




    vox = vox.reshape(shape)

    coord_x=array('i')
    coord_y=array('i')
    coord_z=array('i')
    voxel_r=array('f')
    voxel_g=array('f')
    voxel_b=array('f')

    count = 0

    for x in range(shape[0]):
        for y in range (shape[1]):
             for z in range(shape[2]):
                 if (vox[x,y,z,0]>0) or (vox[x,y,z,1]>0) or (vox[x,y,z,2]) > 0:
                     if sample is None or sample > random.random():
                         coord_x.append(x)
                         coord_y.append(y)
                         coord_z.append(z)
                         voxel_r.append(vox[x,y,z,0]/255)
                         voxel_g.append(vox[x,y,z,1]/255)
                         voxel_b.append(vox[x,y,z,2]/255)
                         count += 1
    print("saving...")
    f = open(name, 'wb')
    f.write(struct.pack("i", count))
    f.write(struct.pack(str(count)+"i", *coord_x))
    f.write(struct.pack(str(count)+"i", *coord_y))
    f.write(struct.pack(str(count)+"i", *coord_z))
    f.write(struct.pack(str(count)+"f", *voxel_r))
    f.write(struct.pack(str(count)+"f", *voxel_g))
    f.write(struct.pack(str(count)+"f", *voxel_b))
    f.close()

    print(count, "done...")



def prediction_export(name, vox, weights, shape, tsdf_range=None):

    from array import array
    import struct



    vox = vox.reshape(shape)

    coord_x=array('i')
    coord_y=array('i')
    coord_z=array('i')
    voxel_v=array('f')

    count = 0

    for x in range(shape[0]):
        for y in range (shape[1]):
             for z in range(shape[2]):
                 if ((vox[x,y,z]!=0) and (weights[x,y,z]!=0)):
                     coord_x.append(x)
                     coord_y.append(y)
                     coord_z.append(z)
                     voxel_v.append(vox[x,y,z])
                     count += 1
    print("saving...")
    f = open(name, 'wb')
    f.write(struct.pack("i", count))
    f.write(struct.pack(str(count)+"i", *coord_x))
    f.write(struct.pack(str(count)+"i", *coord_y))
    f.write(struct.pack(str(count)+"i", *coord_z))
    f.write(struct.pack(str(count)+"f", *voxel_v))
    f.close()

    print(count, "done...")

class_colors = [
    (0.1, 0.1, 0.1),
    (0.0649613, 0.467197, 0.0667303),
    (0.1, 0.847035, 0.1),
    (0.0644802, 0.646941, 0.774265),
    (0.131518, 0.273524, 0.548847),
    (1, 0.813553, 0.0392201),
    (1, 0.490452, 0.0624932),
    (0.657877, 0.0505005, 1),
    (0.0363214, 0.0959549, 0.548847),
    (0.316852, 0.548847, 0.186899),
    (0.548847, 0.143381, 0.0045568),
    (1, 0.241096, 0.718126),
    (0.9, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (0.3, 0.3, 0.3)
    ]

class_names = ["empty", "ceiling", "floor", "wall", "window", "chair", "bed", "sofa",
               "table", "tvs", "furniture", "objects", "error1", "error2", "error3"]


def write_header(obj_file, mtl_file, name):
    obj_file.write("# EdgeNet360 Wavefront obj exporter v1.0\n")
    obj_file.write("mtllib %s.mtl\n" % os.path.basename(name))
    obj_file.write("o Cube\n")

    mtl_file.write("# EdgeNet360 Wavefront obj exporter v1.0\n")
    # Blender MTL File: 'DWRC1.blend'
    # Material Count: 11


def write_vertice(obj_file, x, y, z, cx, cy, cz, v_unit):
    vu = v_unit * 4
    obj_file.write("v %8.6f %8.6f %8.6f\n" %((x-cx)*vu, (y-cy)*vu, (z-cz)*vu))

def write_vertice_normals(obj_file):
    obj_file.write("vn -1.000000  0.000000  0.000000\n")
    obj_file.write("vn  0.000000  0.000000 -1.000000\n")
    obj_file.write("vn  1.000000  0.000000  0.000000\n")
    obj_file.write("vn  0.000000  0.000000  1.000000\n")
    obj_file.write("vn  0.000000 -1.000000  0.000000\n")
    obj_file.write("vn  0.000000  1.000000  0.000000\n")


def write_mtl_faces(obj_file, mtl_file, mtl_faces_list, cl, triangular):
    obj_file.write("g %s\n" % class_names[cl])
    obj_file.write("usemtl %s\n" % class_names[cl])
    #obj_file.write("s  off\n")
    mtl_file.write("newmtl %s\n" % class_names[cl])

    mtl_file.write("Ns 96.078431\n")
    mtl_file.write("Ka 1.000000 1.000000 1.000000\n")
    mtl_file.write("Kd %8.6f %8.6f %8.6f\n" % (class_colors[cl][0], class_colors[cl][1], class_colors[cl][2] ) )
    mtl_file.write("Ks 0.500000 0.500000 0.500000\n")
    mtl_file.write("Ke 0.000000 0.000000 0.000000\n")
    mtl_file.write("Ni 1.000000\n")
    mtl_file.write("d 1.000000\n")
    mtl_file.write("illum 2\n")


    if not triangular:
        for face_vertices in mtl_faces_list:

            obj_file.write("f ")
            obj_file.write("%d//%d  %d//%d %d//%d %d//%d" % (
                                                                face_vertices[1]+1, face_vertices[0],
                                                                face_vertices[2]+1, face_vertices[0],
                                                                face_vertices[3]+1, face_vertices[0],
                                                                face_vertices[4]+1, face_vertices[0],
                                                            ))
            obj_file.write("\n")
    else:
        for face_vertices in mtl_faces_list:

            obj_file.write("f ")
            obj_file.write("%d//%d  %d//%d %d//%d" % (
                                                            face_vertices[1]+1, face_vertices[0],
                                                            face_vertices[2]+1, face_vertices[0],
                                                            face_vertices[3]+1, face_vertices[0],
                                                        ))
            obj_file.write("\n")


def obj_export(name, vox, shape, camx, camy, camz, v_unit, include_top=False, triangular=False, inner_faces=True, material_map = None):

    vu = v_unit * 4

    vox = vox.reshape(shape)

    num_classes=len(class_names)

    sx, sy, sz = vox.shape

    _vox = np.ones((sx+2,sy+2,sz+2), dtype=np.uint8)*255
    _vox[1:-1,1:-1,1:-1] = vox

    cx, cy, cz = int(sx//2), int(camy//(v_unit*4)), int(sz//2)

    vox_ctrl = np.ones((sx+1, sy+1, sz+1), dtype=np.int32) * -1

    mtl_faces_list =[None] * num_classes

    num_vertices = 0

    vertex_coords = np.zeros((1000,1000,1000), dtype='int32')
    vertex_categories = []
    vertex_list = []

    # Create a dictionary to store the mapping from voxel classes to texture IDs
    vox_to_texture = defaultdict(list)
    #Object splitting 
    print("voxel split\n")
    objectCheck = ["window", "chair", "bed", "sofa","table", "tvs", "furniture", "objects"]
    added_class =[]
    added_colors =[]
    for idx,k in enumerate(class_names):
        if k in objectCheck:
            temp = np.zeros(vox.shape)
            for x in range(sx):
                for y in range(sy):
                    for z in range(sz):
                        mtl = int(vox[x,y,z])
                        
                        if mtl == 0 or mtl == 255:
                            continue
                        try:
                            if class_names[mtl] == k:
                                temp[x,y,z] = 1
                        except:
                            continue

            voxel_split_labels = measure.label(temp,connectivity =2)
            numLabels = len(np.unique(voxel_split_labels))-1
            class_color = class_colors[idx]
            for i in range(numLabels):
                added_class.append(str(k+" "+str(i)))
                added_colors.append(class_color)
                
            for x in range(sx):
                for y in range(sy):
                    for z in range(sz):
                        flag = voxel_split_labels[x,y,z]
                        if flag == 0:
                            continue

                        else:
                            vox[x,y,z] = int((len(class_names)+len(added_class))-(numLabels-flag) -1)
    
    class_names.extend(added_class)
    class_colors.extend(added_colors)
    num_classes=len(class_names)
    mtl_faces_list =[None] * num_classes

    # Calculate frequency of each material per object
    for i in range(vox.shape[0]):
        for j in range(vox.shape[1]):
            for k in range(vox.shape[2]):
                voxel_class = vox[i, j, k]
                texture_id = material_map[i, j, k]

                if texture_id != 0:
                    material_key = None  # Initialize with a default value
                    for key, value in grayscale_plate.items():
                        if texture_id == value:
                            material_key = key
                            break
                    if material_key is not None:
                        vox_to_texture[voxel_class].append(material_key)

    majority_texture = {}

    # Assign most common material per object
    for voxel_class, texture_ids in vox_to_texture.items():
        print(class_names[voxel_class],":\n")
        most_common = Counter(texture_ids).most_common()
        for rank, (value, frequency) in enumerate(most_common, 1):
            print(f'Rank: {rank}, Value: {value}, Frequency: {frequency}')
            
        most_common_texture_id, _ = Counter(texture_ids).most_common(1)[0]
        if voxel_class != 0 and voxel_class!=255:
            majority_texture[class_names[voxel_class]] = material_labels.get(most_common_texture_id)


    # find highest point of voxel that is not empty
    highest_point = 0
    for y in range(sy):
        if np.max(vox[:,y,:]) > 0:
            highest_point = y
    print("highest point is ", highest_point)

    # these numbers are found by trial and error/estimation
    ceiling_cutoff = highest_point - 1
    # different cutoff for different outputs apparently
    if name == "./Output/Input_prediction" :
        ceiling_cutoff = highest_point - 16
    elif name == "./Output/Input_prediction_mesh":
        ceiling_cutoff = highest_point - 16
    print("ceiling cutoff is ", ceiling_cutoff)

    with open(name+".obj", 'w') as obj_file, open(name+".mtl", 'w') as mtl_file:

        write_header(obj_file, mtl_file, name)

        for x in range(sx):
            for y in range(sy):
                # old code have y>26 as cutoff point, have no clue where he got this number, doesnt work for all models so I changed it to more dynamic cutoff point
                if not include_top and y>ceiling_cutoff:
                    #print("skipping top at y=",y)
                    continue
                for z in range(sz):
                    mtl = int(vox[x,y,z])
                    if mtl == 0 or mtl==255:
                        continue
                    for vx in range(2):
                        for vy in range(2):
                            for vz in range(2):
                                if vox_ctrl[x+vx, y+vy, z+vz] == -1:
                                    vox_ctrl[x + vx, y + vy, z + vz] = num_vertices
                                    num_vertices += 1
                                    write_vertice(obj_file, x+vx, y+vy, z+vz, cx, cy, cz, v_unit)
                                    x_index = int(x + vx - cx + 500)
                                    y_index = int(y + vy - cy + 500)
                                    z_index = int(z + vz - cz + 500)
                                    vertex_coords[x_index][y_index][z_index] = 1
                                    vertex_categories.append(mtl)
                                    vertex_list.append([(x_index - 500) * vu, (y_index - 500) * vu, (z_index - 500) * vu])
                    if mtl_faces_list[mtl] is None:
                        mtl_faces_list[mtl] = []

                    if inner_faces:

                        if triangular:

                            mtl_faces_list[mtl].extend([
                                [1, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+0, z+1]], #OK
                                [1, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+0, y+0, z+1]], #OK

                                [2, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+0, y+0, z+0]], #OK
                                [2, vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+1, y+0, z+0], vox_ctrl[x+0, y+0, z+0]], #OK

                                [3, vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+0, z+0]], #OK
                                [3, vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+0, z+1], vox_ctrl[x+1, y+0, z+0]], #OK

                                [4, vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+0, z+1]], #OK
                                [4, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+0, y+0, z+1], vox_ctrl[x+1, y+0, z+1]], #OK

                                [5, vox_ctrl[x+0, y+0, z+1], vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+1, y+0, z+1]], #OK
                                [5, vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+1, y+0, z+0], vox_ctrl[x+1, y+0, z+1]], #OK

                                [6, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+1, z+0]], #OK
                                [6, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+1, z+0]]  #OK
                            ])

                        else:

                            mtl_faces_list[mtl].extend([
                                [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                 vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 1]],  # OK
                                [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                 vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 0]],  # OK
                                [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                 vox_ctrl[x + 1, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 0]],  # OK
                                [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                 vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                 vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                 vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 0]]  # OK
                            ])
                    else:
                        _x, _y, _z = x+1, y+1, z+1
                        if triangular:

                            if _vox[_x - 1,_y,_z] != _vox[_x, _y, _z] and _vox[_x-1,_y,_z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 1]],  # OK
                                    [1, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y, _z-1] != _vox[_x, _y, _z] and _vox[_x, _y, _z-1]!=255:
                                mtl_faces_list[mtl].extend([
                                    [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0]],  # OK
                                    [2, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 0, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0]]])  # OK

                            if _vox[_x + 1, _y, _z] != _vox[_x, _y, _z] and _vox[_x + 1, _y, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 0]],  # OK
                                    [3, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 0, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 0]]])  # OK

                            if _vox[_x, _y, _z + 1] != _vox[_x, _y, _z] and _vox[_x, _y, _z + 1]!=255:
                                mtl_faces_list[mtl].extend([
                                    [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                    [4, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 0, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y-1, _z] != _vox[_x, _y, _z] and _vox[_x, _y-1, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                    [5, vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y + 1, _z] != _vox[_x, _y, _z] and _vox[_x, _y + 1, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 0]],  # OK
                                    [6, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 0]]])  # OK

                        else:

                            if _vox[_x - 1, _y, _z] != _vox[_x, _y, _z] and _vox[_x - 1, _y, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y, _z - 1] != _vox[_x, _y, _z] and _vox[_x, _y, _z - 1]!=255:
                                mtl_faces_list[mtl].extend([
                                    [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 0]]])  # OK
                            if _vox[_x + 1, _y, _z] != _vox[_x, _y, _z] and _vox[_x + 1, _y, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 0]]])  # OK
                            if _vox[_x, _y, _z + 1] != _vox[_x, _y, _z] and _vox[_x, _y, _z + 1]!=255:
                                mtl_faces_list[mtl].extend([
                                    [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y - 1, _z] != _vox[_x, _y, _z] and _vox[_x, _y - 1, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y + 1, _z] != _vox[_x, _y, _z] and _vox[_x, _y + 1, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 0]]])  # OK

        write_vertice_normals(obj_file)

        for mtl in range(num_classes):
            if not  mtl_faces_list[mtl] is None:
                write_mtl_faces(obj_file, mtl_file, mtl_faces_list[mtl], mtl, triangular)

        new_object_labels = []
        for k,v in majority_texture.items():
            if k != "floor" and k != "wall" and k!= "ceiling":
                temp = np.zeros(vox.shape)
                for x in range(sx):
                    for y in range(sy):
                        for z in range(sz):
                            mtl = int(vox[x,y,z])
                            if mtl == 0 or mtl == 255:
                                continue
                            if class_names[mtl] == k:
                                temp[x,y,z] = 1

                voxel_split_labels = measure.label(temp,connectivity =2)
                numLabels = np.unique(voxel_split_labels)
                print(k, "split objects: ", len(numLabels))

        
        
        print("Material assigned to object-> ",majority_texture)

        draw_mesh(name+"_mesh.obj", vertex_coords, vertex_categories, vertex_list, v_unit, majority_texture,class_names)

    return

