import bpy
import bmesh
import os
import math

# Global default vars
BASE_ARMPIT_ANGLE = 2.461024985435223
BASE_ELBOW_ANGLE = 2.9540990558848352
BASE_HIP_ANGLE = 1.6520571392932055
BASE_SHOULDER_HIP_DIFF = (108.602844 - 60.33493) / 2
BASE_UPPER_BODY_HEIGHT = 165.0949536
BASE_UPPER_LEG_LENGTH = 95.903725
BASE_LOWER_LEG_LENGTH = 78.31119
BASE_UPPER_ARM_LENGTH = 77.71532
BASE_LOWER_ARM_LENGTH = 67.15182

def load_blend(model_path):
    bpy.ops.wm.open_mainfile(filepath=model_path)

def load_obj(model_path):
    bpy.ops.import_scene.obj(filepath=model_path)
    return bpy.context.selected_objects[-1].name

def load_fbx(model_path):
    bpy.ops.import_scene.fbx(filepath=model_path)
    return bpy.context.selected_objects[-1].name

def select_armature_object():
    armature_object = None
    # Ensure we are in Object Mode
    if bpy.context.object:
        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

    # Loop through objects in the scene to find an armature object
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature_object = obj
            break

    if armature_object:
        bpy.ops.object.select_all(action='DESELECT')

        armature_object.select_set(True)
        bpy.context.view_layer.objects.active = armature_object

        return armature_object.name
    else:
        print("No armature object found in the scene.")
        return None

def select_mesh_object():
    mesh_object = None

    if bpy.context.object:
        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')    

    # Loop through objects in the scene to find a mesh object
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_object = obj
            break

    if mesh_object:
        bpy.ops.object.select_all(action='DESELECT')

        mesh_object.select_set(True)
        bpy.context.view_layer.objects.active = mesh_object

        return mesh_object.name
    else:
        print("No mesh object found in the scene.")
        return None

def create_uv_map(model_name):
    obj = bpy.data.objects[model_name]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.align(axis='ALIGN_S')
    # bpy.ops.uv.smart_project()
    # bpy.ops.uv.unwrap()
    bpy.ops.uv.cube_project(cube_size=1, correct_aspect=False, clip_to_bounds=False, scale_to_bounds=True)
    bpy.ops.object.mode_set(mode='OBJECT')
 
    # obj = bpy.data.objects[model_name]
    # bpy.context.view_layer.objects.active = obj

    # Project view? - Failed
    # if obj.type == 'MESH':
    #     bpy.ops.object.mode_set(mode='EDIT')

    #     # Select all faces of the mesh
    #     bpy.ops.mesh.select_all(action='SELECT')

    #     # Ensure a UV Map exists
    #     if not obj.data.uv_layers:
    #         obj.data.uv_layers.new()

    #     # Project from View
    #     bpy.ops.uv.project_from_view(scale_to_bounds=True)

    #     # Return to Object Mode
    #     bpy.ops.object.mode_set(mode='OBJECT')
    # else:
    #     print(f"The object {model_name} is not a mesh.")

def apply_texture(model_name, texture_path):
    absolute_texture_path = os.path.abspath(texture_path)

    img = bpy.data.images.load(absolute_texture_path)

    mat = bpy.data.materials.new(name="ModelMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = img
    
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    obj = bpy.data.objects[model_name]
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def export_model(export_path):
    if export_path.endswith('.obj'):
      bpy.ops.export_scene.obj(
        filepath=export_path, 
        use_selection=True,
        use_mesh_modifiers=True, 
        use_edges=True, 
        use_smooth_groups=False, 
        use_smooth_groups_bitflags=False, 
        use_normals=True, 
        use_uvs=True, 
        use_materials=True, 
        use_triangles=True, 
        use_nurbs=False, 
        use_vertex_groups=False, 
        use_blen_objects=True, 
        group_by_object=False, 
        group_by_material=False, 
        keep_vertex_order=False, 
        global_scale=1, 
        path_mode='AUTO'
      )
    if export_path.endswith('.glb'):
      bpy.ops.export_scene.gltf(
        filepath=export_path, 
        export_format='GLB', 
        export_animations=True,
        export_animation_mode='SCENE',
        export_apply=True, 
        export_image_format='AUTO', 
        export_texture_dir='textures', 
        export_normals=True, 
        export_tangents=True, 
        export_colors=True, 
        export_cameras=True, 
        export_lights=True, 
        export_extras=True
      )

def calculate_scale(data, type="upper"):
    if type == "upper":
        avg_height = (data["left_shoulder-left_hip"] + data["right_shoulder-right_hip"]) / 2
        return avg_height / BASE_UPPER_BODY_HEIGHT
    if type =="upperarm.L":
        return data["left_shoulder-left_elbow"] / BASE_UPPER_ARM_LENGTH
    if type =="upperarm.R":
        return data["right_shoulder-right_elbow"] / BASE_UPPER_ARM_LENGTH
    if type =="lowerarm.L":
        return data["left_elbow-left_wrist"] / BASE_LOWER_ARM_LENGTH
    if type =="lowerarm.R":
        return data["right_elbow-right_wrist"] / BASE_LOWER_ARM_LENGTH
    if type =="upperleg.L":
        return data["left_hip-left_knee"] / BASE_UPPER_LEG_LENGTH
    if type =="upperleg.R":
        return data["right_hip-right_knee"] / BASE_UPPER_LEG_LENGTH
    if type =="lowerleg.L":
        return data["left_knee-left_ankle"] / BASE_LOWER_LEG_LENGTH
    if type =="lowerleg.R":
        return data["right_knee-right_ankle"] / BASE_LOWER_LEG_LENGTH
    if type == "lat":
        avg_length = (data["left_shoulder-right_shoulder"] - data["left_hip-right_hip"]) / 2
        return avg_length / BASE_SHOULDER_HIP_DIFF
    return 1
    
def adjust_rig(scale_data, angle_data, model_name):
    bpy.context.view_layer.objects.active = bpy.data.objects[model_name]

    if bpy.context.active_object.type == 'ARMATURE':
        # Switch to Pose Mode
        bpy.ops.object.mode_set(mode='POSE')
        armature = bpy.data.objects[model_name]

        # Get the pose bones
        # Adjust lat
        scale_lat = calculate_scale(scale_data, type="lat")
        armature.data.bones['KTF.L'].select = True
        bpy.ops.transform.resize(value=(scale_lat, scale_lat, scale_lat))
        armature.data.bones['KTF.L'].select = False

        armature.data.bones['KTF.R'].select = True
        bpy.ops.transform.resize(value=(scale_lat, scale_lat, scale_lat))
        armature.data.bones['KTF.R'].select = False

        # Adjust arms
        for bone_name in ["upperarm.L", "upperarm.R", "lowerarm.L", "lowerarm.R"]:
            scale = calculate_scale(scale_data, type=bone_name)
            armature.data.bones[bone_name].select = True
            bpy.ops.transform.resize(value=(scale, scale, scale))
            armature.data.bones[bone_name].select = False
        
        # Adjust legs
        for bone_name in ["upperleg.L", "upperleg.R", "lowerleg.L", "lowerleg.R"]:
            scale = calculate_scale(scale_data, type=bone_name)
            armature.data.bones[bone_name].select = True
            bpy.ops.transform.resize(value=(scale, scale, scale))
            armature.data.bones[bone_name].select = False
        
        # Adjust upper body height
        scale_height = calculate_scale(scale_data, type="upper")
        armature.data.bones['chest'].select = True
        bpy.ops.transform.resize(value=(scale_height, scale_height, scale_height))
        armature.data.bones['chest'].select = False

        # Translate waist position from legs' length differences (Max for people with disabilities)
        upperleg_length_diff = max(scale_data["left_hip-left_knee"], scale_data["right_hip-right_knee"]) - BASE_UPPER_LEG_LENGTH
        lowerleg_length_diff = max(scale_data["left_knee-left_ankle"], scale_data["right_knee-right_ankle"]) - BASE_LOWER_LEG_LENGTH
        armature.data.bones['wiest'].select = True
        bpy.ops.transform.translate(value=(0, 0, (upperleg_length_diff + lowerleg_length_diff) / 40))
        armature.data.bones['wiest'].select = False

        angle_key = {
            "upperarm.L": "left_shoulder",
            "upperarm.R": "right_shoulder",
            "lowerarm.L": "left_elbow",
            "lowerarm.R": "right_elbow",
            "upperleg.L": "left_hip",
            "upperleg.R": "right_hip",
        }

        # Adjust arm angles
        for bone_name in ["upperarm.L", "upperarm.R"]:
            angle = angle_data[angle_key[bone_name]] - BASE_ARMPIT_ANGLE
            armature.data.bones[bone_name].select = True
            bpy.ops.transform.rotate(value=angle, orient_axis='Y')
            armature.data.bones[bone_name].select = False
        
        # # Adjust leg angles - The leg joints are locked for some reasons
        # for bone_name in ["upperleg.L", "upperleg.R"]:
        #     angle = angle_data[angle_key[bone_name]] - BASE_HIP_ANGLE
        #     armature.data.bones[bone_name].select = True
        #     bpy.ops.transform.rotate(value=angle, orient_axis='Z')
        #     armature.data.bones[bone_name].select = False

        # Adjust elbow angles
        for bone_name in ["lowerarm.L", "lowerarm.R"]:
            angle = angle_data[angle_key[bone_name]] - BASE_ELBOW_ANGLE
            armature.data.bones[bone_name].select = True
            bpy.ops.transform.rotate(value=angle * 10, orient_axis='Y')
            armature.data.bones[bone_name].select = False
    else:
        print(f"The object {model_name} is not  an armature.")

model_path = './3d/character.blend'
model_path = './3d/character.blend'
def convert2dto3d(texture_path, scale_data, angle_data, export_path):
    load_blend(model_path)

    amature_name = select_armature_object()
    model_name = select_mesh_object()

    adjust_rig(scale_data, angle_data, amature_name)

    create_uv_map(model_name)
    apply_texture(model_name, texture_path)

    export_model(export_path)