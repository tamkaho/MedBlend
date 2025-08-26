"""
MIT License

Copyright (c) 2023 Michael Douglass

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import bpy
import bpy.utils.previews
from bpy_extras.io_utils import ImportHelper, ExportHelper
import openvdb as openvdb
import numpy as np
from pathlib import Path

# Import 3rd party packages from Python Wheels
import pydicom
from platipy.dicom.io.rtstruct_to_nifti import read_dicom_image
from platipy.dicom.io.rtstruct_to_nifti import transform_point_set_from_dicom_struct
import SimpleITK as sitk

# MedBlend Custom Packages
from .proton import is_proton_plan, is_rtplan, has_beam_sequence
from .dicom_util import (
    check_dicom_image_type,
    is_dose_file,
    load_dicom_images,
    sort_by_instance_number,
    extract_dicom_data,
    filter_by_series_uid,
    find_monaco_ct_directory,
)
from .node_groups import (
    apply_DICOM_shader,
    apply_proton_spots_geo_nodes,
    ensure_volume_display_settings,
    verify_volume_material,
    optimize_volume_visibility,
    update_dose_material_thresholds,
)
from .blender_utils import add_data_fields, create_object


# DICOM Spatial Transformation Functions
def get_dicom_spatial_info(ds):
    """
    Extract spatial information from a DICOM dataset.

    Args:
        ds: pydicom Dataset

    Returns:
        Dictionary containing spatial metadata
    """
    spatial_info = {
        "modality": getattr(ds, "Modality", "Unknown"),
        "image_position": getattr(ds, "ImagePositionPatient", None),
        "image_orientation": getattr(ds, "ImageOrientationPatient", None),
        "pixel_spacing": getattr(ds, "PixelSpacing", None),
        "slice_thickness": getattr(ds, "SliceThickness", None),
        "spacing_between_slices": getattr(ds, "SpacingBetweenSlices", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "grid_frame_offsets": getattr(ds, "GridFrameOffsetVector", None),
        "dose_grid_scaling": getattr(ds, "DoseGridScaling", None),
        "dose_units": getattr(ds, "DoseUnits", "Unknown"),
    }

    # Convert to numpy arrays where appropriate
    if spatial_info["image_position"] is not None:
        spatial_info["image_position"] = np.array(
            [float(x) for x in spatial_info["image_position"]]
        )
    if spatial_info["image_orientation"] is not None:
        spatial_info["image_orientation"] = np.array(
            [float(x) for x in spatial_info["image_orientation"]]
        )
    if spatial_info["pixel_spacing"] is not None:
        spatial_info["pixel_spacing"] = np.array(
            [float(x) for x in spatial_info["pixel_spacing"]]
        )
    if spatial_info["grid_frame_offsets"] is not None:
        spatial_info["grid_frame_offsets"] = np.array(
            [float(x) for x in spatial_info["grid_frame_offsets"]]
        )

    return spatial_info


def calculate_volume_coordinates(spatial_info, pixel_array_shape=None):
    """
    Calculate 3D coordinate arrays based on DICOM spatial information.

    Args:
        spatial_info: Dictionary from get_dicom_spatial_info()
        pixel_array_shape: Optional shape tuple (z, y, x) for 3D volumes

    Returns:
        Dictionary containing coordinate arrays and bounds
    """
    image_pos = spatial_info["image_position"]
    pixel_spacing = spatial_info["pixel_spacing"]
    rows = spatial_info["rows"]
    columns = spatial_info["columns"]
    grid_offsets = spatial_info["grid_frame_offsets"]

    if image_pos is None or pixel_spacing is None or rows is None or columns is None:
        raise ValueError(
            "Missing required spatial parameters for coordinate calculation"
        )

    # Calculate X and Y coordinate arrays (DICOM coordinate system)
    # DICOM ImagePositionPatient is the center of the top-left pixel of the image
    # X coordinates: image_position[0] + i * pixel_spacing[0]  (column direction)
    # Y coordinates: image_position[1] + j * pixel_spacing[1]  (row direction)

    # Note: DICOM PixelSpacing is [row_spacing, column_spacing]
    row_spacing = pixel_spacing[0]  # Y direction spacing
    column_spacing = pixel_spacing[1]  # X direction spacing

    # Create coordinate arrays for each voxel center
    x_coords = image_pos[0] + np.arange(columns) * column_spacing
    y_coords = image_pos[1] + np.arange(rows) * row_spacing

    # Z coordinates depend on modality and available information
    z_coords = None
    if grid_offsets is not None:
        # For RTDOSE: Z = image_position[2] + grid_frame_offset
        # Grid frame offsets are relative to the image position
        z_coords = image_pos[2] + grid_offsets
    elif pixel_array_shape is not None and len(pixel_array_shape) >= 3:
        # For CT/MR: estimate Z coordinates based on slice count
        num_slices = pixel_array_shape[2]
        spacing_between_slices = spatial_info.get("spacing_between_slices", None)
        if spacing_between_slices is None:
            spacing_between_slices = spatial_info.get("slice_thickness", 1.0)
        z_coords = image_pos[2] + np.arange(num_slices) * spacing_between_slices
    else:
        # Single slice: just the image position Z
        z_coords = np.array([image_pos[2]])

    # Ensure Z coordinates are sorted (important for proper alignment)
    if len(z_coords) > 1:
        z_coords = np.sort(z_coords)

    coords = {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "x_range": [x_coords.min(), x_coords.max()],
        "y_range": [y_coords.min(), y_coords.max()],
        "z_range": [z_coords.min(), z_coords.max()]
        if z_coords is not None
        else [image_pos[2], image_pos[2]],
        "image_position": image_pos,
        "pixel_spacing": pixel_spacing,
    }

    return coords


def create_blender_transform_matrix(coords):
    """
    Create a Blender transformation matrix for proper DICOM positioning.

    Args:
        coords: Dictionary from calculate_volume_coordinates()
        volume_shape: Shape of the volume (z, y, x)

    Returns:
        4x4 transformation matrix for OpenVDB/Blender
    """
    pixel_spacing = coords["pixel_spacing"]

    # Get voxel dimensions in mm
    voxel_size_x = pixel_spacing[1]  # Column spacing (DICOM X direction)
    voxel_size_y = pixel_spacing[0]  # Row spacing (DICOM Y direction)

    # For Z, use the spacing between z coordinates
    z_coords = coords["z_coords"]
    if len(z_coords) > 1:
        voxel_size_z = abs(z_coords[1] - z_coords[0])  # Spacing between slices
    else:
        voxel_size_z = coords.get("slice_thickness", 1.0) or 1.0

    # Convert to meters for Blender (Blender uses meters as default unit)
    scale_x = voxel_size_x / 1000.0  # mm to m
    scale_y = voxel_size_y / 1000.0  # mm to m
    scale_z = abs(voxel_size_z) / 1000.0  # mm to m, ensure positive

    # Create transformation matrix
    # OpenVDB grid transformation maps voxel indices to world coordinates
    # For DICOM:
    #   - X increases left to right
    #   - Y increases posterior to anterior
    #   - Z increases inferior to superior
    # This matches Blender's coordinate system
    transform = np.array(
        [
            [scale_x, 0, 0, 0],
            [0, scale_y, 0, 0],
            [0, 0, scale_z, 0],
            [0, 0, 0, 1],
        ]
    )

    return transform


def calculate_volume_position_blender(coords):
    """
    Calculate the position offset for the Blender volume object.

    Args:
        coords: Dictionary from calculate_volume_coordinates()

    Returns:
        3D position vector in meters for Blender object location
    """
    # DICOM ImagePositionPatient refers to the center of the first voxel (upper-left-back corner)
    # But OpenVDB/Blender volume transform + object location should place volume correctly

    # Use the minimum coordinates of the volume (corner of the volume bounding box)
    # This ensures the volume is positioned at the right location in world space
    x_min = coords["x_range"][0]
    y_min = coords["y_range"][0]
    z_min = coords["z_range"][0]

    # Convert DICOM position (mm) to Blender position (meters)
    position = np.array(
        [
            x_min / 1000.0,  # X: left-right
            y_min / 1000.0,  # Y: anterior-posterior
            z_min / 1000.0,  # Z: inferior-superior
        ]
    )

    return position


# A function to display custom messages to the user
def show_message_box(message="", title="Message Box", icon="INFO"):
    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


addon_keymaps = {}
_icons = None


class SNA_PT_MEDBLEND_70A7C(bpy.types.Panel):
    bl_label = "MedBlend"
    bl_idname = __package__
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = ""
    bl_category = "Medical"
    bl_order = 0
    bl_ui_units_x = 0

    @classmethod
    def poll(cls, context):
        return not (False)

    def draw_header(self, context):
        layout = self.layout

    def draw(self, context):
        layout = self.layout

        # Only displays the load buttons if the required dependancies are installed

        layout.label(text="Images", icon_value=125)
        op = layout.operator(
            "medblend.load_ct",
            text="Load DICOM Images",
            icon_value=0,
            emboss=True,
            depress=False,
        )
        layout.label(text="Dose", icon_value=851)
        op = layout.operator(
            "medblend.load_dose",
            text="Load DICOM Dose",
            icon_value=0,
            emboss=True,
            depress=False,
        )
        layout.label(text="Structures", icon_value=304)
        op = layout.operator(
            "medblend.load_structures",
            text="Load DICOM Structures",
            icon_value=0,
            emboss=True,
            depress=False,
        )
        layout.label(text="Proton Spots", icon_value=653)
        op = layout.operator(
            "medblend.load_proton",
            text="Load Proton Plan",
            icon_value=0,
            emboss=True,
            depress=False,
        )


# Class to load CT or MRI Images
class SNA_OT_Load_Ct_Fc7B9(bpy.types.Operator, ImportHelper):
    bl_idname = "medblend.load_ct"
    bl_label = "Load CT"
    bl_description = "Load a CT Dataset"
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: bpy.props.StringProperty(default="*.dcm", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        if bpy.app.version >= (3, 0, 0) and True:
            cls.poll_message_set("")
        return not False

    def execute(self, context):
        file_name_CT = self.filepath
        print("The name of the CT is " + str(file_name_CT))

        selected_file = pydicom.dcmread(file_name_CT)
        if check_dicom_image_type(selected_file):
            series_uid = selected_file.SeriesInstanceUID

            dir_path = Path(file_name_CT)
            dir_path = dir_path.parents[0]

            # Load all DICOM CT images from your folder using load_dicom_images function
            images = load_dicom_images(dir_path)
            # Filter out only those DICOM CT images that have your specified series UID using filter_by_series_uid function
            filtered_images = filter_by_series_uid(images, series_uid)

            # Sort those filtered DICOM CT slices by their instance number using sort_by_instance_number function
            sorted_images = sort_by_instance_number(filtered_images)

            (
                CT_volume,
                spacing,
                slice_position,
                slice_spacing,
                image_origin,
                image_orientation,
                image_columns,
            ) = extract_dicom_data(sorted_images)

            # NEW: Calculate proper spatial transformation using DICOM spatial utilities
            print("=== Calculating proper DICOM spatial transformation ===")

            # Get spatial info from first dataset
            first_ds = sorted_images[0]
            spatial_info = get_dicom_spatial_info(first_ds)

            # Override with actual z positions from all slices
            z_positions = []
            for ds in sorted_images:
                z_pos = float(ds.ImagePositionPatient[2])
                z_positions.append(z_pos)

            # Calculate coordinate system
            pixel_array_shape = CT_volume.shape
            coords = calculate_volume_coordinates(spatial_info, pixel_array_shape)

            # Override Z coordinates with actual slice positions
            coords["z_coords"] = np.array(sorted(z_positions))
            coords["z_range"] = [min(z_positions), max(z_positions)]

            # Create Blender transformation matrix
            transform_matrix = create_blender_transform_matrix(coords)
            volume_position = calculate_volume_position_blender(coords)

            # Create an OpenVDB volume from the pixel data
            grid = openvdb.FloatGrid()

            # Copies image volume from numpy to VDB grid
            grid.copyFromArray(CT_volume.astype(float))
            print(
                f"CT Volume dimensions: {CT_volume.shape}; Transformation Matrix: f{transform_matrix}"
            )

            # Convert numpy array to OpenVDB-compatible format (4x4 matrix)
            vdb_transform = openvdb.createLinearTransform(transform_matrix.tolist())
            grid.transform = vdb_transform

            # Sets the grid class to FOG_VOLUME
            grid.gridClass = openvdb.GridClass.FOG_VOLUME

            # Blender needs grid name to be "Density"
            grid.name = "density"

            # Writes CT volume to a vdb file but perhaps this could be done internally in the future
            openvdb.write(str(dir_path.joinpath("CT.vdb")), grid)
            print(f"Writing to {str(dir_path.joinpath('CT.vdb'))}")

            # Add the volume to the scene
            bpy.ops.object.volume_import(
                filepath=str(dir_path.joinpath("CT.vdb")), files=[]
            )

            # Set the volume's position using proper DICOM positioning
            if bpy.context.active_object and bpy.context.active_object.type == "VOLUME":
                volume_obj = bpy.context.active_object
                volume_obj.location = volume_position

                # Store spatial metadata for alignment verification
                volume_obj["spatial_coords"] = {
                    "x_range": coords["x_range"],
                    "y_range": coords["y_range"],
                    "z_range": coords["z_range"],
                    "image_position": coords["image_position"].tolist(),
                    "pixel_spacing": coords["pixel_spacing"].tolist(),
                    "modality": spatial_info["modality"],
                }

                print(f"Set volume position: {volume_position} m")
                print(f"Stored spatial metadata in volume object")
        else:
            print("No DICOM images loaded")

        # Apply material and ensure proper display settings
        apply_DICOM_shader("Image Material")
        ensure_volume_display_settings()
        verify_volume_material("Image Material")
        optimize_volume_visibility()

        return {"FINISHED"}


# Class to load Proton Plan files
class SNA_OT_Load_Proton_1Dbc6(bpy.types.Operator, ImportHelper):
    bl_idname = "medblend.load_proton"
    bl_label = "Load Proton"
    bl_description = "Load Proton Spots and Weights"
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: bpy.props.StringProperty(default="*.dcm", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        if bpy.app.version >= (3, 0, 0) and True:
            cls.poll_message_set("")
        return not False

    def execute(self, context):
        file_name_proton = self.filepath

        dataset = pydicom.dcmread(file_name_proton)

        if is_proton_plan(dataset):
            print("File is proton plan")
            return self._load_proton_plan(dataset)
        elif is_rtplan(dataset) and has_beam_sequence(dataset):
            print("File is standard RT plan with beam sequence")
            return self._load_rtplan(dataset)
        else:
            print("File is not a supported plan type")
            show_message_box(
                "This file is not a supported RT plan type. Please select a proton plan or standard RT plan.",
                "Unsupported Plan Type",
                "ERROR",
            )
            return {"CANCELLED"}

    def _load_proton_plan(self, dataset):
        """Load proton plan with IonBeamSequence"""
        import math

        pi_value = math.pi

        BeamNo = 0
        for beam in dataset.IonBeamSequence:
            control_points = beam.IonControlPointSequence
            num_control_points = len(control_points)
            # frame_index = 1
            # bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
            # empty = bpy.data.objects['Empty']
            spots = []
            x = []
            y = []
            E = []
            spot_weights = []
            for i in range(0, num_control_points, 2):
                spots_in_energy_layer = len(
                    beam.IonControlPointSequence[i].ScanSpotPositionMap
                )
                weights = control_points[i].ScanSpotMetersetWeights
                for j in range(0, spots_in_energy_layer, 2):
                    x.append(control_points[i].ScanSpotPositionMap[j] / 1000)

                    y.append(control_points[i].ScanSpotPositionMap[j + 1] / 1000)

                    E.append(float(control_points[i].NominalBeamEnergy) / 1000)

                    spot_weights.append(weights[int(j / 2)])

                    # spots.append((x,y,E))
                    # spot_weights.append((W,0,0))
                    # print(x,y,E)
                    # if weights[int(j/2)]>0:
                    # empty.location = (x,y,E)
                    # empty.keyframe_insert(data_path="location", frame=frame_index)
                    # frame_index=frame_index + 1
                    # bpy.ops.mesh.primitive_uv_sphere_add(location=(x,y,E), radius=weights[int(j/2)]/10)
            # print(spots)
            # print(spot_weights)
            # print(np.shape(spot_weights))
            # print(np.shape(spots))
            gantry_angle = float(beam.IonControlPointSequence[0].GantryAngle)
            iso_center = (
                np.asarray(beam.IonControlPointSequence[0].IsocenterPosition) / 1000
            )
            # mesh = bpy.data.meshes.new('ProtonSpots')
            # mesh.from_pydata(spots, [], [])
            # obj = bpy.data.objects.new('ProtonSpots', mesh)

            # bpy.context.scene.collection.objects.link(obj)
            # obj.rotation_euler[1] = gantry_angle*pi_value/180
            # obj.location[0] = iso_center[2]
            # obj.location[1] = iso_center[1]
            # obj.location[2] = iso_center[0]

            # mesh = bpy.data.meshes.new('ProtonWeights')
            # mesh.from_pydata(spot_weights, [], [])
            # obj = bpy.data.objects.new('ProtonWeights', mesh)

            # bpy.context.scene.collection.objects.link(obj)
            mesh = bpy.data.meshes.new(name="proton_spots")

            data_fields = ["spot_x", "spot_y", "spot_E", "spot_weight"]
            add_data_fields(mesh, data_fields)

            for row in range(0, np.shape(spot_weights)[0]):
                mesh.vertices.add(1)
                mesh.update()  # might be slow, but does it matter?...
                # assign row values to mesh attribute values
                for data_field in data_fields:
                    mesh.attributes["spot_x"].data[row].value = x[row]
                    mesh.attributes["spot_y"].data[row].value = y[row]
                    mesh.attributes["spot_E"].data[row].value = E[row]
                    mesh.attributes["spot_weight"].data[row].value = spot_weights[row]
                mesh.vertices[row].co = (
                    0.01 * row,
                    0.0,
                    0.0,
                )  # set vertex x position according to index

            mesh.update()
            mesh.validate()

            # create object if data was imported
            if len(mesh.vertices) > 0:
                obj = create_object(mesh, [f"proton_spots{BeamNo}"][0])

            obj.rotation_euler[1] = gantry_angle * pi_value / 180
            obj.location[0] = iso_center[0]
            obj.location[1] = iso_center[1]
            obj.location[2] = iso_center[2]

            BeamNo = BeamNo + 1
            apply_proton_spots_geo_nodes(node_tree_name="Proton_Spots")

        # print(np.shape(weights))
        return {"FINISHED"}

    def _load_rtplan(self, dataset):
        """Load standard RT plan with BeamSequence (Monaco format)"""
        import math

        pi_value = math.pi

        print(f"Loading RT Plan: {dataset.get('RTPlanLabel', 'Unknown')}")

        BeamNo = 0
        for beam in dataset.BeamSequence:
            print(f"Processing Beam {BeamNo + 1}")

            # Get beam parameters
            beam_name = beam.get("BeamName", f"Beam_{BeamNo + 1}")
            beam_type = beam.get("BeamType", "STATIC")
            treatment_machine = beam.get("TreatmentMachineName", "Unknown")

            # Get control points - RT plans use ControlPointSequence
            if hasattr(beam, "ControlPointSequence"):
                control_points = beam.ControlPointSequence

                # Get beam geometry from first control point
                if len(control_points) > 0:
                    first_cp = control_points[0]
                    gantry_angle = float(first_cp.get("GantryAngle", 0))
                    collimator_angle = float(first_cp.get("BeamLimitingDeviceAngle", 0))

                    # Get isocenter position
                    iso_center = np.array([0, 0, 0])  # Default if not available
                    if hasattr(first_cp, "IsocenterPosition"):
                        iso_center = np.asarray(first_cp.IsocenterPosition) / 1000

                    # Get dose rate if available
                    dose_rate = first_cp.get("DoseRateSet", 600)  # Default 600 MU/min

                    print(f"  Beam: {beam_name}")
                    print(f"  Gantry Angle: {gantry_angle}°")
                    print(f"  Collimator Angle: {collimator_angle}°")
                    print(f"  Isocenter: {iso_center}")
                    print(f"  Dose Rate: {dose_rate} MU/min")

                    # Create a simple beam representation
                    # For now, create an empty object to represent the beam
                    bpy.ops.object.empty_add(type="SINGLE_ARROW", align="WORLD")
                    beam_obj = bpy.context.active_object
                    beam_obj.name = f"RTBeam_{beam_name}_{BeamNo}"

                    # Set beam orientation and position
                    beam_obj.rotation_euler[1] = (
                        gantry_angle * pi_value / 180
                    )  # Gantry rotation
                    beam_obj.rotation_euler[2] = (
                        collimator_angle * pi_value / 180
                    )  # Collimator rotation
                    beam_obj.location = iso_center
                    beam_obj.scale = (0.1, 0.1, 0.1)  # Make it smaller

                    # Add custom properties for beam data
                    beam_obj["beam_name"] = beam_name
                    beam_obj["beam_type"] = beam_type
                    beam_obj["treatment_machine"] = treatment_machine
                    beam_obj["gantry_angle"] = gantry_angle
                    beam_obj["collimator_angle"] = collimator_angle
                    beam_obj["dose_rate"] = dose_rate

                    # Get MLC positions if available
                    if hasattr(first_cp, "BeamLimitingDevicePositionSequence"):
                        mlc_data = []
                        for device in first_cp.BeamLimitingDevicePositionSequence:
                            device_type = device.get(
                                "RTBeamLimitingDeviceType", "Unknown"
                            )
                            if hasattr(device, "LeafJawPositions"):
                                positions = device.LeafJawPositions
                                mlc_data.append(
                                    {"type": device_type, "positions": positions}
                                )
                        beam_obj["mlc_data"] = str(mlc_data)  # Store as string

            BeamNo += 1

        print(f"Loaded {BeamNo} beams from RT Plan")
        return {"FINISHED"}


# Class to load DICOM Dose files
class SNA_OT_Load_Dose_7629F(bpy.types.Operator, ImportHelper):
    bl_idname = "medblend.load_dose"
    bl_label = "Load Dose"
    bl_description = "Load a DICOM Dose File"
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: bpy.props.StringProperty(default="*.dcm", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        if bpy.app.version >= (3, 0, 0) and True:
            cls.poll_message_set("")
        return not False

    def execute(self, context):
        # DICOM spatial functions are now integrated in this file (no import needed)

        file_name_dose = self.filepath

        # Directory containing DICOM images
        DICOM_dir = file_name_dose

        file_path = Path(DICOM_dir)
        root_dir = file_path.parents[0]

        # Load the DICOM dataset
        ds = pydicom.read_file(DICOM_dir)
        if is_dose_file(ds):
            # Get the Dose Grid
            pixel_data = ds.pixel_array
            pixel_data = np.rot90(pixel_data, k=-1, axes=(0, 2))
            pixel_data = np.ascontiguousarray(pixel_data)

            # Get Dose Grid Scaling (3004,000E) - crucial for correct dose values
            dose_grid_scaling = 1.0  # Default fallback
            if hasattr(ds, "DoseGridScaling"):
                dose_grid_scaling = float(ds.DoseGridScaling)
                print(f"Found Dose Grid Scaling: {dose_grid_scaling}")
            else:
                raise ValueError("Dose Grid Scaling not found in DICOM dose file")

            # NEW: Calculate proper spatial transformation using DICOM spatial utilities
            print("=== Calculating proper DICOM dose spatial transformation ===")

            # Get spatial info from dose datasetextract_dicom_data
            spatial_info = get_dicom_spatial_info(ds)

            # Calculate coordinate system for dose
            pixel_array_shape = pixel_data.shape
            coords = calculate_volume_coordinates(spatial_info, pixel_array_shape)

            # Create Blender transformation matrix
            transform_matrix = create_blender_transform_matrix(coords)
            volume_position = calculate_volume_position_blender(coords)

            # Converts list to numpy array and apply dose grid scaling
            dose_matrix = np.asarray(pixel_data, dtype=float)
            dose_matrix = np.flipud(dose_matrix)

            # Apply DICOM Dose Grid Scaling to get actual dose values in Gy
            dose_matrix = dose_matrix * dose_grid_scaling + 0.1  # remove the +0.1 later

            print(f"Raw pixel range: {pixel_data.min()} to {pixel_data.max()}")
            print(
                f"Scaled dose range: {dose_matrix.min():.4f} to {dose_matrix.max():.4f} Gy"
            )
            print(f"Dose units: {getattr(ds, 'DoseUnits', 'Unknown')}")

            # Create an OpenVDB volume from the pixel data
            # Creates a grid of Float precision
            grid = openvdb.FloatGrid()
            print(
                f"Dose Volume dimensions: {dose_matrix.shape}, Transformation Matrix: {transform_matrix}"
            )

            # Copies image volume from numpy to VDB grid
            grid.copyFromArray(dose_matrix.astype(float))

            # Apply proper DICOM-based transformation matrix
            print(f"Applying transformation matrix:")
            print(
                f"  Scale: [{transform_matrix[0, 0] * 1000:.3f}, {transform_matrix[1, 1] * 1000:.3f}, {transform_matrix[2, 2] * 1000:.3f}] mm/voxel"
            )

            # Convert numpy array to OpenVDB-compatible format (4x4 matrix)
            vdb_transform = openvdb.createLinearTransform(transform_matrix.tolist())
            grid.transform = vdb_transform

            # Sets the grid class to FOG_VOLUME
            grid.gridClass = openvdb.GridClass.FOG_VOLUME
            # Blender needs grid name to be "Density"
            grid.name = "density"

            # Create unique VDB filename based on source DICOM file to prevent conflicts
            source_filename = file_path.stem  # Get filename without extension
            unique_vdb_name = f"dose_{source_filename}.vdb"
            dose_dir = root_dir.joinpath(unique_vdb_name)
            print(f"Creating unique VDB file: {unique_vdb_name}")

            # Writes CT volume to a vdb file but perhaps this could be done internally in the future
            openvdb.write(str(dose_dir), grid)

            # Add the volume to the scene
            bpy.ops.object.volume_import(filepath=str(dose_dir), files=[])

            # Set the volume's position using proper DICOM positioning
            if bpy.context.active_object and bpy.context.active_object.type == "VOLUME":
                dose_obj = bpy.context.active_object
                dose_obj.location = volume_position

                # Store dose metadata in the volume object for material shader use
                dose_obj["dose_min"] = float(dose_matrix.min())
                dose_obj["dose_max"] = float(dose_matrix.max())
                dose_obj["dose_grid_scaling"] = dose_grid_scaling
                dose_obj["dose_units"] = getattr(ds, "DoseUnits", "Unknown")
                dose_obj["source_dicom"] = source_filename  # Track source file

                # Store spatial metadata for alignment verification
                dose_obj["spatial_coords"] = {
                    "x_range": coords["x_range"],
                    "y_range": coords["y_range"],
                    "z_range": coords["z_range"],
                    "image_position": coords["image_position"].tolist(),
                    "pixel_spacing": coords["pixel_spacing"].tolist(),
                    "modality": spatial_info["modality"],
                }

                print(f"Set dose volume position: {volume_position} m")
                print(f"Stored dose metadata in volume object: {dose_obj.name}")

        else:
            print("No Dose File Loaded")

        # Create unique material name based on filename or object name
        unique_material_name = "Dose Material"
        if bpy.context.active_object and bpy.context.active_object.type == "VOLUME":
            dose_obj = bpy.context.active_object
            # Create unique material name based on source DICOM filename
            source_name = dose_obj.get("source_dicom", dose_obj.name)
            from .node_groups import create_short_unique_material_name

            unique_material_name = create_short_unique_material_name(source_name)
            print(f"Using unique material name: {unique_material_name}")

        # Apply material with unique name and ensure proper display settings
        actual_material_name = apply_DICOM_shader(unique_material_name)
        ensure_volume_display_settings()
        verify_volume_material(actual_material_name)
        optimize_volume_visibility()

        # Update dose material thresholds with actual dose values using the actual material name (handles truncation)
        if bpy.context.active_object and bpy.context.active_object.type == "VOLUME":
            dose_obj = bpy.context.active_object
            if "dose_min" in dose_obj and "dose_max" in dose_obj:
                update_dose_material_thresholds(
                    dose_obj["dose_min"], dose_obj["dose_max"], actual_material_name
                )

        return {"FINISHED"}


# Class to load DICOM Structure files as volumes
class SNA_OT_Load_Structures_5Ebc9(bpy.types.Operator, ImportHelper):
    bl_idname = "medblend.load_structures"
    bl_label = "Load Structures"
    bl_description = "Load a DICOM Structure Set"
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: bpy.props.StringProperty(default="*.dcm", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        if bpy.app.version >= (3, 0, 0) and True:
            cls.poll_message_set("")
        return not False

    def execute(self, context):
        from pathlib import Path
        # DICOM spatial functions are now integrated in this file (no import needed)

        structure_path = self.filepath

        # Assume structure_file_path is the path to the structure file specified by the user
        structure_file_path = Path(structure_path)

        # Get the directory of the structure file
        structure_directory = structure_file_path.parent
        directory_path = structure_directory

        # Try to find Monaco CT directory structure first
        monaco_ct_dir = find_monaco_ct_directory(structure_path)
        if monaco_ct_dir:
            print(f"Using Monaco CT directory: {monaco_ct_dir}")
            directory_path = Path(monaco_ct_dir)
        else:
            print(f"Using default directory (same as RT struct): {directory_path}")

        print(f"Structure path: {structure_path}")
        print(f"CT directory path: {directory_path}")

        try:
            DICOM_IMAGE = read_dicom_image(directory_path)
            voxel_resolution = DICOM_IMAGE.GetSpacing()
            origin = DICOM_IMAGE.GetOrigin()
        except Exception as e:
            print(f"Error reading DICOM image from {directory_path}: {e}")
            show_message_box(
                f"Error reading CT images from directory: {directory_path}. Please check the file structure.",
                "DICOM Reading Error",
                "ERROR",
            )
            return {"CANCELLED"}

        dicom_structure = pydicom.dcmread(structure_path)

        try:
            struct_masks, struct_names = transform_point_set_from_dicom_struct(
                DICOM_IMAGE, dicom_structure
            )

        except Exception as e:
            print(f"Error transforming structure data: {e}")
            show_message_box(
                "Something is wrong with the structure file. Please check the file and try again.",
                "Error",
                "ERROR",
            )
            return {"CANCELLED"}

        # NEW: Get spatial info for proper transformation alignment
        print("=== Calculating proper DICOM structure spatial transformation ===")

        ct_files = list(directory_path.glob("*.dcm"))
        if ct_files:
            sample_ct = pydicom.dcmread(ct_files[0])
            ct_spatial_info = get_dicom_spatial_info(sample_ct)
            ct_spatial_info["image_position"] = np.array(DICOM_IMAGE.GetOrigin())
            ct_volume_shape = DICOM_IMAGE.GetSize()

            # Calculate coordinate system for CT (which the structures are aligned to)
            ct_coords = calculate_volume_coordinates(ct_spatial_info, ct_volume_shape)

        else:
            print("Warning: Could not find CT files for spatial verification")
            ct_coords = None

        # Create transformation matrix that matches the CT coordinate system
        transform_matrix = create_blender_transform_matrix(ct_coords)
        volume_position = calculate_volume_position_blender(ct_coords)

        for i in range(0, len(struct_masks)):
            numpy_image = sitk.GetArrayFromImage(struct_masks[i])
            numpy_image = np.rot90(numpy_image, k=-1, axes=(2, 1))
            numpy_image = np.ascontiguousarray(np.transpose(numpy_image, (1, 2, 0)))

            print("Structure Name:", struct_names[i])
            print("Structure Shape:", np.shape(numpy_image))
            print("Structure Voxel Resolution:", voxel_resolution)
            print("Structure Origin:", origin)

            # Creates a grid of Double precision
            grid = openvdb.FloatGrid()
            # Copies image volume from numpy to VDB grid
            grid.copyFromArray(numpy_image.astype(float))

            print(
                f"Using DICOM-aligned transformation for structure: {struct_names[i]}"
            )
            print(
                f"  Transform scale: [{transform_matrix[0, 0] * 1000:.3f}, {transform_matrix[1, 1] * 1000:.3f}, {transform_matrix[2, 2] * 1000:.3f}] mm/voxel"
            )

            # Convert numpy array to OpenVDB-compatible format (4x4 matrix)
            vdb_transform = openvdb.createLinearTransform(transform_matrix.tolist())
            grid.transform = vdb_transform

            # Sets the grid class to FOG_VOLUME
            grid.gridClass = openvdb.GridClass.FOG_VOLUME
            # Blender needs grid name to be "Density"
            grid.name = "density"

            struct_dir = structure_directory.joinpath("structs.vdb")
            struct_dir = structure_directory.joinpath(f"{struct_names[i]}.vdb")
            # Writes CT volume to a vdb file but perhaps this could be done internally in the future
            openvdb.write(str(struct_dir), grid)

            # Add the volume to the scene
            bpy.ops.object.volume_import(filepath=str(struct_dir), files=[])

            # Set the volume's position using proper DICOM positioning
            if bpy.context.active_object and bpy.context.active_object.type == "VOLUME":
                struct_obj = bpy.context.active_object
                struct_obj.location = volume_position

                # Store spatial metadata if we have CT coords
                if ct_coords is not None:
                    struct_obj["spatial_coords"] = {
                        "x_range": ct_coords["x_range"],
                        "y_range": ct_coords["y_range"],
                        "z_range": ct_coords["z_range"],
                        "image_position": ct_coords["image_position"].tolist(),
                        "pixel_spacing": ct_coords["pixel_spacing"].tolist(),
                        "modality": "RTSTRUCT",
                    }

                print(f"Set structure volume position: {volume_position} m")

            # Apply material and ensure proper display settings
            apply_DICOM_shader("Structure Material")
            ensure_volume_display_settings()
            verify_volume_material("Structure Material")
            optimize_volume_visibility()

        return {"FINISHED"}


def register():
    global _icons
    _icons = bpy.utils.previews.new()
    bpy.utils.register_class(SNA_PT_MEDBLEND_70A7C)
    bpy.utils.register_class(SNA_OT_Load_Ct_Fc7B9)
    bpy.utils.register_class(SNA_OT_Load_Proton_1Dbc6)
    bpy.utils.register_class(SNA_OT_Load_Dose_7629F)
    bpy.utils.register_class(SNA_OT_Load_Structures_5Ebc9)


def unregister():
    global _icons
    bpy.utils.previews.remove(_icons)
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    for km, kmi in addon_keymaps.values():
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    bpy.utils.unregister_class(SNA_PT_MEDBLEND_70A7C)
    bpy.utils.unregister_class(SNA_OT_Load_Ct_Fc7B9)
    bpy.utils.unregister_class(SNA_OT_Load_Proton_1Dbc6)
    bpy.utils.unregister_class(SNA_OT_Load_Dose_7629F)
    bpy.utils.unregister_class(SNA_OT_Load_Structures_5Ebc9)


# register()
