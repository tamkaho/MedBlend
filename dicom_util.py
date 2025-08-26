# dicom_utilities.py

import os

import numpy as np

try:
    import pydicom

except ImportError:
    print(
        "pydicom is not installed. Please install it using the command 'pip install pydicom'"
    )

    pydicom = None


# Checks if the DICOM file is a RTDose file.
def is_dose_file(ds):
    """
    Checks if the DICOM file at the given path is of type dose.
    Returns True if it is, False otherwise.
    """
    try:
        if ds.Modality == "RTDOSE":
            return True
        else:
            return False
    except:
        return False


# Checks if the DICOM file is a RTStructure file.
def is_structure_file(ds):
    """
    Checks if the DICOM file at the given path is of type structure.
    Returns True if it is, False otherwise.
    """
    try:
        if ds.Modality == "RTSTRUCT":
            return True
        else:
            return False
    except:
        return False


# Checks if the DICOM file is a CT or MRI image
def check_dicom_image_type(ds):
    """
    Check if a DICOM file is a CT or MRI image
    :param dicom_file_path: path to the DICOM file
    :return: 'CT' if the file is a CT image, 'MRI' if the file is an MRI image, and 'Unknown' if the file is neither CT nor MRI
    """
    try:
        if ds.Modality == "CT":
            return 1
        elif ds.Modality == "MR":
            return 1
        else:
            return 0

    except Exception as e:
        print(f"Error: {e}")
        return "Unknown"


# Define a function to load DICOM images from a folder
def load_dicom_images(folder):
    """
    Load DICOM images from a folder and return a list of the images
    :param folder: path to the folder containing the DICOM images
    :return: list of DICOM images
    """
    import pydicom

    # Create an empty list to store the images
    images = []
    # Loop through all files in the folder
    for filename in os.listdir(folder):
        # Check if the file is a DICOM file
        if filename.endswith(".dcm"):
            # Read the file using pydicom
            image = pydicom.dcmread(os.path.join(folder, filename))
            if image and check_dicom_image_type(image):
                # Append the image to the list
                images.append(image)
    # Return the list of images
    return images


def find_monaco_ct_directory(rtstruct_path):
    """
    Find the CT or MR directory for Monaco DICOM data structure.
    Monaco organizes files in sibling directories:
    - RT struct: .../RTSTRUCT_*/series_*/
    - CT images: .../CT_*/series_*/ OR MR images: .../MR_*/series_*/

    :param rtstruct_path: Path to the RT struct file
    :return: Path to the directory containing CT/MR DICOM images, or None if not found
    """
    from pathlib import Path

    try:
        # Get the RT struct file path as a Path object
        struct_path = Path(rtstruct_path)

        # Navigate up to the main directory (e.g., tst0)
        # Structure: .../tst0/RTSTRUCT_*/series_*/file.dcm
        # We want to go up 3 levels: file -> series -> RTSTRUCT -> tst0
        main_dir = struct_path.parent.parent.parent

        # Look for CT or MR directory pattern
        for item in main_dir.iterdir():
            if item.is_dir() and (
                item.name.startswith("CT_") or item.name.startswith("MR_")
            ):
                # Found CT/MR directory, now find the series subdirectory
                for series_dir in item.iterdir():
                    if series_dir.is_dir() and series_dir.name.startswith("series_"):
                        # Check if this directory contains DCM files
                        dcm_files = list(series_dir.glob("*.dcm"))
                        if dcm_files:
                            print(f"Found Monaco CT/MR directory: {series_dir}")
                            return str(series_dir)

        print(
            f"Warning: Could not find CT/MR directory for Monaco structure at {rtstruct_path}"
        )
        return None

    except Exception as e:
        print(f"Error finding Monaco CT/MR directory: {e}")
        return None


# Define a function to sort DICOM images by instance number
def sort_by_instance_number(images):
    """
    Sort the list of images by their instance number attribute using lambda function
    :param images: list of DICOM images
    :return: sorted list of DICOM images
    """
    # Sort the list of images by their instance number attribute
    sorted_images = sorted(images, key=lambda x: x.InstanceNumber)
    # Return the sorted list of images
    return sorted_images


def extract_dicom_data(images):
    dicom_3d_array = []
    slice_position = []
    for i in range(0, len(images)):
        dicom_3d_array.append(images[i].pixel_array)
        slice_position.append(images[i].ImagePositionPatient)

    dicom_3d_array = np.asarray(dicom_3d_array)
    dicom_3d_array = np.rot90(dicom_3d_array, k=-1, axes=(2, 1))
    dicom_3d_array = np.ascontiguousarray(np.transpose(dicom_3d_array, (1, 2, 0)))
    return (
        dicom_3d_array,
        images[0].PixelSpacing,
        slice_position,
        images[0].SliceThickness,
        images[0].ImagePositionPatient,
        images[0].ImageOrientationPatient,
        images[0].Columns,
    )


def filter_by_series_uid(images, series_uid):
    """
    Filter a list of DICOM images by their series instance UID
    :param images: list of DICOM images
    :param series_uid: series instance UID to filter by
    :return: filtered list of DICOM images
    """

    # Create an empty list to store the filtered images
    filtered_images = []
    # Loop through all images in the list
    for image in images:
        # Check if the image has the same series UID as specified
        if image.SeriesInstanceUID == series_uid:
            # Append the image to the filtered list
            filtered_images.append(image)
    # Return the filtered list of images
    return filtered_images
