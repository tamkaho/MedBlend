import bpy
import os


def append_item_from_blend(file_path, item_type, item_name):
    """
    Append an item from a .blend file
    """
    # Use os.path.join for cross-platform compatibility
    directory_path = os.path.join(file_path, item_type, "")
    bpy.ops.wm.append(directory=directory_path, filename=item_name)


def create_short_unique_material_name(source_name, base_name="Dose Material"):
    """
    Create a shorter unique material name to avoid Blender's truncation issues.

    :param source_name: Original source filename (DICOM file stem)
    :param base_name: Base material name
    :return: Shortened unique material name that fits within Blender's limits
    """
    max_length = 60  # Conservative limit to avoid truncation

    # If the full name fits, use it
    full_name = f"{base_name} - {source_name}"
    if len(full_name) <= max_length:
        return full_name

    # Create a shorter unique identifier
    import hashlib

    # Use last 8 characters + hash for uniqueness
    if len(source_name) > 16:
        # Extract meaningful parts: first few chars + last 8 chars + hash
        prefix = source_name[:6] if len(source_name) > 6 else source_name
        suffix = source_name[-8:] if len(source_name) > 8 else source_name

        # Create short hash for uniqueness
        hash_obj = hashlib.md5(source_name.encode())
        short_hash = hash_obj.hexdigest()[:6]

        short_identifier = f"{prefix}...{suffix}_{short_hash}"
    else:
        # Short enough, just add hash if needed
        hash_obj = hashlib.md5(source_name.encode())
        short_hash = hash_obj.hexdigest()[:6]
        short_identifier = f"{source_name}_{short_hash}"

    short_name = f"{base_name} - {short_identifier}"

    # Ensure it still fits
    if len(short_name) > max_length:
        # Further truncate if needed
        available_chars = max_length - len(base_name) - 3  # 3 for " - "
        hash_obj = hashlib.md5(source_name.encode())
        unique_id = hash_obj.hexdigest()[:available_chars]
        short_name = f"{base_name} - {unique_id}"

    print(f"Shortened material name: '{full_name}' → '{short_name}'")
    print(f"Length: {len(full_name)} → {len(short_name)} chars")

    return short_name


def apply_DICOM_shader(shader_name):
    current_path = bpy.path.abspath(os.path.dirname(__file__))

    # combine current path with the immediate sub-folder called "assets"
    path = os.path.join(current_path, "assets")
    # sets assets file name to MedBlend_Assets.blend
    assets_file = os.path.join(path, "MedBlend_Assets.blend")

    # Check if this is a unique material name (contains " - ")
    if " - " in shader_name:
        # Extract base material name (e.g., "Dose Material" from "Dose Material - dose.001")
        base_material_name = shader_name.split(" - ")[0]
        print(
            f"Detected unique material request: '{shader_name}' based on '{base_material_name}'"
        )
    else:
        base_material_name = shader_name

    # Always ensure we have a fresh base material for dose materials
    if base_material_name == "Dose Material":
        # Remove existing base material to ensure fresh copy from assets
        if base_material_name in bpy.data.materials:
            print(
                f"Removing existing base material to get fresh copy: '{base_material_name}'"
            )
            bpy.data.materials.remove(
                bpy.data.materials[base_material_name], do_unlink=True
            )

    # Ensure the base material exists (load from assets if needed)
    if base_material_name not in bpy.data.materials:
        print(f"Loading base material '{base_material_name}' from assets")
        append_item_from_blend(assets_file, "Material", base_material_name)

    # Track the actual material name (which may be truncated by Blender)
    actual_material_name = shader_name

    # Check if the requested material (unique or base) already exists
    if shader_name not in bpy.data.materials:
        if shader_name != base_material_name:
            # Create a copy of the base material with the unique name
            print(f"Creating unique material copy: '{shader_name}'")
            base_material = bpy.data.materials[base_material_name]
            unique_material = base_material.copy()

            # Assign the name - Blender may truncate long names
            unique_material.name = shader_name
            actual_material_name = (
                unique_material.name
            )  # Capture the actual name (may be truncated)

            # Log if truncation occurred
            if actual_material_name != shader_name:
                print("WARNING: Material name truncated by Blender:")
                print(f"  Requested: '{shader_name}'")
                print(f"  Actual:    '{actual_material_name}'")

            # The material copy should already have an independent node tree
            # But let's ensure it's properly set up for nodes
            if not unique_material.use_nodes:
                unique_material.use_nodes = True

            print(f"Created unique material: '{actual_material_name}'")
        # If shader_name == base_material_name, the base material was already loaded above
    else:
        # Material already exists, use its current name
        actual_material_name = shader_name
        print(f"Using existing material: '{actual_material_name}'")

    # Use the actual material name for lookup (handles truncation)
    target_material = bpy.data.materials[actual_material_name]
    bpy.context.object.data.materials.append(target_material)
    print(
        f"Applied material '{target_material.name}' to object '{bpy.context.object.name}'"
    )

    # Set viewport shading to Material for better volume visualization
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    # Set shading to Material for volumes
                    if space.shading.type in ["SOLID", "WIREFRAME"]:
                        space.shading.type = "MATERIAL"
                        print(
                            f"Changed viewport shading to Material for {actual_material_name}"
                        )
                    break
            break

    return actual_material_name


def update_dose_material_thresholds(dose_min, dose_max, material_name="Dose Material"):
    """
    Update the dose material shader with proper min/max thresholds based on actual dose values
    This ensures each dose file gets properly calibrated material settings.

    :param dose_min: Minimum dose value in Gy
    :param dose_max: Maximum dose value in Gy
    :param material_name: Name of the dose material to update
    """
    print(
        f"Updating dose material thresholds: {dose_min:.4f} to {dose_max:.4f} Gy for '{material_name}'"
    )

    # Get the dose material
    if material_name not in bpy.data.materials:
        print(f"Warning: Material '{material_name}' not found")
        return False

    material = bpy.data.materials[material_name]

    if not material.use_nodes:
        print(f"Warning: Material '{material_name}' does not use nodes")
        return False

    # Find and update all relevant node types
    nodes_updated = 0

    # Look for Value nodes with exact names that Blender uses for dose materials

    for node in material.node_tree.nodes:
        for input_socket in node.inputs:
            if input_socket.name == "Max Dose" and not input_socket.is_linked:
                input_socket.default_value = dose_max
                print(
                    f"Updated input '{input_socket.name}' on node '{node.name}' to: {dose_max:.4f}"
                )
                nodes_updated += 1
            elif input_socket.name == "Min Dose" and not input_socket.is_linked:
                input_socket.default_value = dose_min
                print(
                    f"Updated input '{input_socket.name}' on node '{node.name}' to: {dose_min:.4f}"
                )
                nodes_updated += 1
            elif input_socket.name == "Intensity" and not input_socket.is_linked:
                input_socket.default_value = 5.0
                print(
                    f"Updated input '{input_socket.name}' on node '{node.name}' to: 5.0"
                )
                nodes_updated += 1

    if nodes_updated == 0:
        print(
            f"Warning: No nodes were updated in material '{material_name}'. Check material setup."
        )
    else:
        print(
            f"Successfully updated {nodes_updated} nodes in material '{material_name}'"
        )

    print(
        f"Dose material '{material_name}' calibrated for range {dose_min:.4f} to {dose_max:.4f} Gy"
    )
    return True


def apply_proton_spots_geo_nodes(node_tree_name="Proton_Spots"):
    """
    Adds the proton spots node group to the selected object
    """
    current_path = bpy.path.abspath(os.path.dirname(__file__))
    # combine current path with the immediate sub-folder called "assets"
    path = os.path.join(current_path, "assets")
    # sets assets file name to MedBlend_Assets.blend
    assets_file = os.path.join(path, "MedBlend_Assets.blend")
    # Check if the node tree exists in the data
    if node_tree_name not in bpy.data.node_groups:
        # Append the node tree from the blend file
        append_item_from_blend(assets_file, "NodeTree", node_tree_name)

    # Get the currently selected object
    obj = bpy.context.active_object

    # Check if the object has a geometry nodes modifier
    geomod = obj.modifiers.get("GeometryNodes")
    if not geomod:
        # If not, create one
        geomod = obj.modifiers.new("GeometryNodes", "NODES")

    # Assign the node tree to the modifier
    geomod.node_group = bpy.data.node_groups[node_tree_name]
