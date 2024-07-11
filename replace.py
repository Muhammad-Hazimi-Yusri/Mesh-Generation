"""
This script does submeshing and splits the meshes based on materials assigned
"""
def split_mesh(obj_file):
    
    with open(obj_file, 'r') as file:
        lines = file.readlines()
        
    new_lines = []
    current_group = None

    for line in lines:
      if line.startswith('usemtl'):
        #get the material name
        material_name = line.strip().split(' ')[1]
        # replace mat with g(gropup) mat to split
        group = material_name.replace('mat', 'g mat')
        #insert a new line if material changes
        if group != current_group:
            new_lines.append(f'{group}\n')
            current_group = group
      new_lines.append(line)

    with open(obj_file, 'w') as file:
        file.writelines(new_lines)

split_mesh("C:\Project\AVVR-Pipeline-Internship\edgenet360\Output\Input_prediction_mesh.obj")
#split_mesh("C:\Project\AVVR-Pipeline-Internship\edgenet360\Data\stanford_processed\\room3\\area_3_2b70fafb2e6f40f193a3d912ff7e5cbe_office_5_EdgeNet_prediction_mesh.obj")