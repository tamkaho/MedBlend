import bpy
import os


def append_item_from_blend(file_path, item_type, item_name):
    """
    Append an item from a .blend file
    """
    # Use os.path.join for cross-platform compatibility
    directory_path = os.path.join(file_path, item_type, "")
    bpy.ops.wm.append(
        directory=directory_path,
        filename=item_name
    )

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

    #combine current path with the immediate sub-folder called "assets"
    path = os.path.join(current_path, "assets")
    #sets assets file name to MedBlend_Assets.blend
    assets_file = os.path.join(path, "MedBlend_Assets.blend")

    # Check if this is a unique material name (contains " - ")
    if " - " in shader_name:
        # Extract base material name (e.g., "Dose Material" from "Dose Material - dose.001")
        base_material_name = shader_name.split(" - ")[0]
        print(f"Detected unique material request: '{shader_name}' based on '{base_material_name}'")
    else:
        base_material_name = shader_name
    
    # Always ensure we have a fresh base material for dose materials
    if base_material_name == "Dose Material":
        # Remove existing base material to ensure fresh copy from assets
        if base_material_name in bpy.data.materials:
            print(f"Removing existing base material to get fresh copy: '{base_material_name}'")
            bpy.data.materials.remove(bpy.data.materials[base_material_name], do_unlink=True)
    
    # Ensure the base material exists (load from assets if needed)
    if not base_material_name in bpy.data.materials:
        print(f"Loading base material '{base_material_name}' from assets")
        append_item_from_blend(assets_file, "Material", base_material_name)
    
    # Track the actual material name (which may be truncated by Blender)
    actual_material_name = shader_name
    
    # Check if the requested material (unique or base) already exists
    if not shader_name in bpy.data.materials:
        if shader_name != base_material_name:
            # Create a copy of the base material with the unique name
            print(f"Creating unique material copy: '{shader_name}'")
            base_material = bpy.data.materials[base_material_name]
            unique_material = base_material.copy()
            
            # Assign the name - Blender may truncate long names
            unique_material.name = shader_name
            actual_material_name = unique_material.name  # Capture the actual name (may be truncated)
            
            # Log if truncation occurred
            if actual_material_name != shader_name:
                print(f"WARNING: Material name truncated by Blender:")
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
    print(f"Applied material '{target_material.name}' to object '{bpy.context.object.name}'")
    
    # Set viewport shading to Material for better volume visualization
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    # Set shading to Material for volumes
                    if space.shading.type in ['SOLID', 'WIREFRAME']:
                        space.shading.type = 'MATERIAL'
                        print(f"Changed viewport shading to Material for {actual_material_name}")
                    break
            break

    return actual_material_name  # Return the actual material name for further use

def ensure_volume_display_settings():
    """
    Ensure proper Blender settings for volume display
    """
    # Enable volume rendering in viewport if available
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    # Ensure we're in a shading mode that shows materials
                    if space.shading.type in ['SOLID', 'WIREFRAME']:
                        space.shading.type = 'MATERIAL'
                        print("Set viewport to Material for volume display")
                    
                    # Enable volume rendering in shading settings
                    if hasattr(space.shading, 'use_scene_lights_render'):
                        space.shading.use_scene_lights_render = True
                    if hasattr(space.shading, 'use_scene_world_render'):
                        space.shading.use_scene_world_render = True
                        
                    break
            break
    
    # Ensure render engine supports volumes
    if bpy.context.scene.render.engine in ['BLENDER_EEVEE', 'CYCLES']:
        print(f"Using {bpy.context.scene.render.engine} render engine - volumes supported")
    else:
        print(f"Warning: {bpy.context.scene.render.engine} may not fully support volume rendering")

def verify_volume_material(material_name):
    """
    Verify that a volume material has proper node setup
    """
    if material_name in bpy.data.materials:
        material = bpy.data.materials[material_name]
        
        if not material.use_nodes:
            material.use_nodes = True
            print(f"Enabled nodes for material: {material_name}")
        
        # Check if material has volume output connected
        has_volume_output = False
        output_node = None
        
        for node in material.node_tree.nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node
                # Check if Volume input is connected
                if node.inputs['Volume'].is_linked:
                    has_volume_output = True
                    print(f"Material {material_name} has volume output connected")
                break
        
        if not has_volume_output and output_node:
            print(f"Warning: Material {material_name} may not have proper volume shader setup")
            
            # Try to add a basic Principled Volume shader if none exists
            volume_nodes = [n for n in material.node_tree.nodes if n.type in ['VOLUME_SCATTER', 'PRINCIPLED_VOLUME']]
            if not volume_nodes:
                print(f"Adding Principled Volume shader to {material_name}")
                principled_volume = material.node_tree.nodes.new(type='ShaderNodeVolumePrincipled')
                material.node_tree.links.new(
                    principled_volume.outputs['Volume'], 
                    output_node.inputs['Volume']
                )
        
        return True
    else:
        print(f"Warning: Material {material_name} not found")
        return False

def optimize_volume_visibility():
    """
    Optimize volume visibility settings for better display in Material mode
    """
    obj = bpy.context.object
    if obj and obj.type == 'VOLUME':
        # Set volume display settings for better visibility
        if hasattr(obj.data, 'display'):
            # Use wireframe display for volume bounds
            obj.data.display.use_slice = False
            obj.data.display.wireframe_type = 'BOXES'
            obj.data.display.wireframe_detail = 'COARSE'
            
        # Adjust material settings for better viewport display
        if obj.data.materials:
            for material in obj.data.materials:
                if material and material.use_nodes:
                    for node in material.node_tree.nodes:
                        if node.type == 'PRINCIPLED_VOLUME':
                            # Increase density for better visibility in viewport
                            if 'Density' in node.inputs:
                                current_density = node.inputs['Density'].default_value
                                if current_density < 0.1:  # If very low, increase it
                                    node.inputs['Density'].default_value = min(current_density * 10, 1.0)
                                    print(f"Increased volume density from {current_density} to {node.inputs['Density'].default_value}")
        
        print("Optimized volume visibility settings")
        return True
    return False

def update_dose_material_thresholds(dose_min, dose_max, material_name="Dose Material"):
    """
    Update the dose material shader with proper min/max thresholds based on actual dose values
    This ensures each dose file gets properly calibrated material settings.
    
    :param dose_min: Minimum dose value in Gy
    :param dose_max: Maximum dose value in Gy  
    :param material_name: Name of the dose material to update
    """
    print(f"Updating dose material thresholds: {dose_min:.4f} to {dose_max:.4f} Gy for '{material_name}'")
    
    # Get the dose material
    if material_name not in bpy.data.materials:
        print(f"Warning: Material '{material_name}' not found")
        return False
    
    material = bpy.data.materials[material_name]
    
    if not material.use_nodes:
        print(f"Warning: Material '{material_name}' does not use nodes")
        return False
    
    # Store dose range information in the material for reference
    material["dose_min"] = dose_min
    material["dose_max"] = dose_max
    material["dose_range"] = dose_max - dose_min
    print(f"Stored dose metadata in material: min={dose_min:.4f}, max={dose_max:.4f}, range={dose_max-dose_min:.4f}")
    
    # Find and update all relevant node types
    nodes_updated = 0
    
    # Find ColorRamp nodes (common for dose visualization)
    color_ramp_nodes = [node for node in material.node_tree.nodes if node.type == 'VALTORGB']
    
    for ramp_node in color_ramp_nodes:
        print(f"Updating ColorRamp node: {ramp_node.name}")
        
        # Reset ColorRamp for this dose's specific range
        if len(ramp_node.color_ramp.elements) >= 2:
            # Clear existing elements beyond the first two
            while len(ramp_node.color_ramp.elements) > 2:
                ramp_node.color_ramp.elements.remove(ramp_node.color_ramp.elements[-1])
            
            # Set up basic two-point ramp: 0 (transparent) to 1 (full dose)
            ramp_node.color_ramp.elements[0].position = 0.0
            ramp_node.color_ramp.elements[0].color = (0, 0, 0, 0)  # Transparent for no dose
            
            ramp_node.color_ramp.elements[1].position = 1.0  
            ramp_node.color_ramp.elements[1].color = (1, 0, 0, 1)  # Red for max dose
            
            # Add middle point for dose threshold visualization
            if dose_max > 0:
                middle_element = ramp_node.color_ramp.elements.new(0.1)  # 10% of max dose
                middle_element.color = (0, 0, 1, 0.3)  # Blue, semi-transparent
                
            print(f"Reset ColorRamp for dose range 0 to {dose_max:.2f} Gy")
            nodes_updated += 1
    
    # Find and update Math nodes used for thresholding
    math_nodes = [node for node in material.node_tree.nodes if node.type == 'MATH']
    
    for math_node in math_nodes:
        if math_node.operation in ['GREATER_THAN', 'LESS_THAN', 'MULTIPLY', 'DIVIDE']:
            # Update threshold values based on this dose's range
            if math_node.operation in ['GREATER_THAN', 'LESS_THAN']:
                # Set threshold to 5% of max dose to filter out low dose noise
                threshold_value = dose_max * 0.05
                if len(math_node.inputs) > 1 and not math_node.inputs[1].is_linked:
                    math_node.inputs[1].default_value = threshold_value
                    print(f"Updated Math node '{math_node.name}' threshold to {threshold_value:.4f}")
                    nodes_updated += 1
            
            elif math_node.operation == 'DIVIDE':
                # Normalize by max dose to map dose values to 0-1 range
                if len(math_node.inputs) > 1 and not math_node.inputs[1].is_linked:
                    # Use max dose as divisor to normalize
                    divisor = dose_max if dose_max > 0 else 1.0
                    math_node.inputs[1].default_value = divisor
                    print(f"Updated Math node '{math_node.name}' divisor to {divisor:.4f} for normalization")
                    nodes_updated += 1
    
    # Find and update Value nodes
    value_nodes = [node for node in material.node_tree.nodes if node.type == 'VALUE']
    
    for value_node in value_nodes:
        node_name_lower = value_node.name.lower()
        
        # Update based on node name keywords
        if any(keyword in node_name_lower for keyword in ['dose', 'threshold', 'min', 'max', 'scale']):
            if 'min' in node_name_lower:
                value_node.outputs[0].default_value = dose_min
                print(f"Updated Value node '{value_node.name}' to dose_min: {dose_min:.4f}")
                nodes_updated += 1
            elif 'max' in node_name_lower:
                value_node.outputs[0].default_value = dose_max  
                print(f"Updated Value node '{value_node.name}' to dose_max: {dose_max:.4f}")
                nodes_updated += 1
            elif 'threshold' in node_name_lower:
                threshold_value = dose_max * 0.05  # 5% threshold
                value_node.outputs[0].default_value = threshold_value
                print(f"Updated Value node '{value_node.name}' to threshold: {threshold_value:.4f}")
                nodes_updated += 1
            elif 'scale' in node_name_lower:
                # Use inverse of max dose for normalization
                scale_value = 1.0 / dose_max if dose_max > 0 else 1.0
                value_node.outputs[0].default_value = scale_value
                print(f"Updated Value node '{value_node.name}' to scale: {scale_value:.6f}")
                nodes_updated += 1
    
    # Look for any Principled Volume nodes and adjust density
    volume_nodes = [node for node in material.node_tree.nodes if node.type == 'PRINCIPLED_VOLUME']
    for volume_node in volume_nodes:
        # Adjust density multiplier based on dose range for better visibility
        if 'Density' in volume_node.inputs:
            # Scale density inversely with dose range for consistent appearance
            base_density = 1.0
            if dose_max > 0:
                # Normalize density: higher max dose = lower density multiplier
                density_multiplier = min(50.0 / dose_max, 10.0)  # Cap at 10x multiplier
                if not volume_node.inputs['Density'].is_linked:
                    volume_node.inputs['Density'].default_value = base_density * density_multiplier
                    print(f"Updated Principled Volume density to {base_density * density_multiplier:.4f}")
                    nodes_updated += 1
    
    if nodes_updated == 0:
        print(f"Warning: No nodes were updated in material '{material_name}'. Check material setup.")
        # List all nodes for debugging
        print("Available nodes in material:")
        for node in material.node_tree.nodes:
            print(f"  - {node.type}: {node.name}")
    else:
        print(f"Successfully updated {nodes_updated} nodes in material '{material_name}'")
    
    print(f"Dose material '{material_name}' calibrated for range {dose_min:.4f} to {dose_max:.4f} Gy")
    return True

def apply_proton_spots_geo_nodes(node_tree_name = 'Proton_Spots'):
    """
    Adds the proton spots node group to the selected object
    """
    current_path = bpy.path.abspath(os.path.dirname(__file__))    
    #combine current path with the immediate sub-folder called "assets"
    path = os.path.join(current_path, "assets")
    #sets assets file name to MedBlend_Assets.blend
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
        geomod = obj.modifiers.new("GeometryNodes", 'NODES')
    
    # Assign the node tree to the modifier
    geomod.node_group = bpy.data.node_groups[node_tree_name]


