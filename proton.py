import bpy
import os
from . constants import ADDON_DIR

def is_proton_plan(ds):
    """
    Checks if the DICOM file at the given path is of type dose.
    Returns True if it is, False otherwise.
    """
    try:
        if ds.Modality == 'RTIon':
            return True
        else:
            return False
    except:
        return False

def is_rtplan(ds):
    """
    Checks if the DICOM file is a standard RT Plan (like Monaco).
    Returns True if it is, False otherwise.
    """
    try:
        if ds.Modality == 'RTPLAN':
            return True
        else:
            return False
    except:
        return False


def has_beam_sequence(ds):
    """
    Checks if the RT Plan has BeamSequence (standard RT plans like Monaco).
    Returns True if it has BeamSequence, False otherwise.
    """
    try:
        if hasattr(ds, 'BeamSequence') and len(ds.BeamSequence) > 0:
            return True
        else:
            return False
    except:
        return False

