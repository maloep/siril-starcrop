#!/usr/bin/env python3
#
# StarCrop
#
# Reference star based sequence cropping for Siril
#
# Copyright (C) 2026
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#

"""
StarCrop

Version : 0.1.0-alpha1

Current status

✓ Connection to Siril
✓ Sequence validation
✓ Reference star
✓ Crop region
✓ Tracking framework

Target Siril version:
    >= 1.4.3
"""

from dataclasses import dataclass
from typing import Optional

from sirilpy import (
    SirilInterface,
    SirilConnectionError,
)


import sirilpy as s
s.ensure_installed("astroalign", "numpy")

import astroalign
import math
import numpy as np


class Config:
    """
    Global configuration.
    """

    VERSION = "0.1.0-alpha1"

    DEBUG = True

    SEARCH_RADII = [
        20,
        40,
        80,
        120,
        150,
    ]

    GRID_STEP = 100

    GRID_OFFSETS = [

        (-150,   0),
        ( 150,   0),

        (   0,-150),
        (   0, 150),

        (-150,-150),
        ( 150,-150),

        (-150, 150),
        ( 150, 150),
    ]

    MAX_MAGNITUDE_DELTA = 0.5
    MAX_MAGNITUDE = 4.8

    MIN_SNR = 8.0

    MAX_FWHMX = 10.0
    MAX_FWHMY = 10.0

    
    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------

    OUTPUT_PREFIX = "starcrop_"
    

class Log:

    @staticmethod
    def info(text=""):
        print(text)

    @staticmethod
    def warning(text):
        print(f"WARNING: {text}")

    @staticmethod
    def error(text):
        print(f"ERROR: {text}")

    @staticmethod
    def header(text):
        print()
        print("=" * 60)
        print(text)
        print("=" * 60)
        

class Sequence:
    """
    Wrapper around the currently loaded Siril sequence.
    """
    
    @property
    def width(self):
        return self.seq.rx


    @property
    def height(self):
        return self.seq.ry 


    def __init__(self, siril: SirilInterface):
        self.siril = siril
        self.seq = siril.get_seq()

        if self.seq is None:
            raise RuntimeError("No sequence loaded.")

        self.name = self.seq.seqname    
        self.current = self.seq.current        
        self.number = self.seq.number
        self.reference_image = self.seq.reference_image
        self.extension: Optional[str] = None


    def detect_extension(self):
        """
        Determine the image extension (.fit/.fits/.fts).

        This method should only be called AFTER all reference
        information has been stored because it temporarily loads
        another frame.
        """

        if self.extension is not None:
            return self.extension

        frame = self.current + 1

        for extension in (".fit", ".fits", ".fts"):

            try:
                self.siril.cmd(
                    f"load {self.name}{frame:05d}{extension}"
                )
                self.extension = extension
                return extension
            except Exception:
                pass

        raise RuntimeError(
            "Unable to determine image extension."
        )


    def load(self, frame_index: int):
        """
        Load a frame of the current sequence.
        """

        if self.extension is None:
            self.detect_extension()

        self.siril.cmd(
            f"load {self.name}{frame_index + 1:05d}{self.extension}"
        )
        
        self.current = frame_index

    def first_frame(self):
        return self.seq.beg - 1


    def last_frame(self):
        return self.seq.end - 1


    def frame_count(self):
        return self.number
        
    
    def save(self, frame_index):
        """
        Save the currently loaded image.
        """

        filename = (
            f"{Config.OUTPUT_PREFIX}"
            f"{frame_index + 1:05d}"
            f"{self.extension}"
        )

        Log.info(f"Saving {filename}")

        self.siril.cmd(f"save {filename}")
        

@dataclass
class TrackedStar:
    """
    Represents a star
    """

    xpos: float
    ypos: float
    mag: float
    SNR: float
    fwhmx: float
    fwhmy: float
    

    @classmethod
    def from_psf_star(cls, star):
        """
        Create a TrackedStar from a Siril PSFStar object.
        """

        return cls(
            xpos=star.xpos,
            ypos=star.ypos,
            mag=star.mag,
            SNR=star.SNR,
            fwhmx=star.fwhmx,
            fwhmy=star.fwhmy,
        )

    def magnitude_difference(self, star):

        return abs(self.mag - star.mag)

    def fwhm_difference(self, star):

        return max(
            abs(self.fwhmx - star.fwhmx),
            abs(self.fwhmy - star.fwhmy),
        )

    
    @classmethod
    def is_valid(cls, star):
        """
        Validate a detected star.
        """


        if not star.phot_is_valid:
            return False, "phot_is_valid == False"

        if not math.isfinite(star.mag):
            return False, "Magnitude is NaN"

        if not math.isfinite(star.SNR):
            return False, "SNR is NaN"

        if not math.isfinite(star.fwhmx):
            return False, "FWHMx is NaN"

        if not math.isfinite(star.fwhmy):
            return False, "FWHMy is NaN"

        if star.mag > Config.MAX_MAGNITUDE:
            return (
                False,
                f"Mag={star.mag:.2f} "
                f"(>{Config.MAX_MAGNITUDE})",
            )
        
        if star.SNR < Config.MIN_SNR:
            return (
                False,
                f"SNR={star.SNR:.2f} "
                f"(<{Config.MIN_SNR})",
            )
            
        if star.fwhmx > Config.MAX_FWHMX:
            return (
                False,
                f"FWHMX={star.fwhmx:.2f} "
                f"(>{Config.MAX_FWHMX})",
            )
            
        if star.fwhmy > Config.MAX_FWHMY:
            return (
                False,
                f"FWHMY={star.fwhmy:.2f} "
                f"(>{Config.MAX_FWHMY})",
            )

        return True, ""
        
    
    @classmethod
    def to_log_string(cls, star):
        return f"""X = {star.xpos}, Y = {star.ypos}, 
            Mag = {star.mag}, FWHMX = {star.fwhmx}, 
            FWHMY = {star.fwhmy}, SNR = {star.SNR}"""
        

class Tracker:

    def __init__(
        self,
        siril,
        sequence,
        selection,
    ):

        self.siril = siril
        self.sequence = sequence
        self.selection = selection


    def track_sequence(self, reference_xy):
        """
        Track the reference star through the whole sequence.
        """

        while self.track_next_frame(reference_xy):
            pass

        Log.header("Tracking finished")
    
    
    def track_next_frame(self, reference_xy):
        #
        # Next frame
        #

        next_frame = self.sequence.current + 1

        if next_frame >= self.sequence.frame_count():
            Log.info("Last frame reached.")
            return False

        Log.header(f"Frame {next_frame}")

        #
        # Load frame
        #

        self.sequence.load(next_frame)        
        Log.info(f"Loaded frame {self.sequence.current}")
        Log.info("Frame loaded.")

        #
        # Search
        #

        #found_star = self.find_star()
        source_pos, target_pos = self.find_star_from_catalog(reference_xy)

        if source_pos is None or target_pos is None:
            Log.warning("no reference star found for current frame.")
            return True
        else:            
            Log.info(
                f"Found star: "
                f"({target_pos[0]:.2f}, "
                f"{target_pos[1]:.2f})"
            )

        
        success = self.crop_current_frame(source_pos, target_pos)        
        
        if(success):
            self.sequence.save(next_frame)

        return True
        
        
    def find_star_from_catalog(self, reference_xy):
        
        sBuilder = StarCatalogBuilder(self.siril)
        
        matches = sBuilder.build_matches_from_reference(reference_xy)
        
        matches_x = []
        matches_y = []
        
        for refpos, matchpos in matches.items():
            if(matchpos is not None):
                matches_x.append((refpos[0], refpos[1]))
                matches_y.append((matchpos[0], matchpos[1]))
        
        Log.info(f"matches_x = {matches_x}")
        Log.info(f"matches_y = {matches_y}")
        
        if(len(matches_x) == 0 or len(matches_y) == 0):
            Log.info("No matches found for current frame")
            return None, None
        
        if(len(matches) == 0):
            Log.info("No matches found")
            return None, None
        
        transform = astroalign.estimate_transform(
            'affine',
            matches_x,
            matches_y
        )
        
        Log.info(f"{transform}")
        
        #return 1st match
        for source, target in matches.items():
            return source, target

        return None, None        
        
        
    def crop_current_frame(self, source_pos, target_pos):
        """
        Crop the currently loaded image.
        """
        crop = CropRegion.from_selection(
            self.selection,
            source_pos
        )

        command = crop.crop_command(target_pos[0], target_pos[1])

        Log.info(command)

        try:
            self.siril.cmd(command)
        except Exception as e:
            Log.warning(f"Error while cropping frame: {e.message}")
            return False

        Log.info("Image cropped.")
        return True
 
 
class StarCatalogBuilder:

    TILE_SIZE = 100
    TILE_STEP = 75

    def __init__(self, siril):

        self.siril = siril


    def build_catalog(self, crop, reference_star=None):

        Log.info("StarCatalogBuilder.build_catalog")
        
        stars = []

        width = crop.width
        height = crop.height
        y = crop.top
        
        if (reference_star is not None):
            stars.append(reference_star) 

        while y < height +crop.top:

            x = crop.left

            while x < width +crop.left:

                shape = self.make_shape(
                    x,
                    y,
                    width +crop.left,
                    height +crop.top
                )

                star = self.find_star(shape)

                if star is not None:
                    stars.append(star)

                x += self.TILE_STEP

            y += self.TILE_STEP

        stars = self.remove_duplicates(stars)

        Log.info("")
        Log.info(f"Catalog contains {len(stars)} stars.")

        return stars
        
        
    def build_matches_from_reference(self, reference_xy):
        
        Log.info("StarCatalogBuilder.build_catalog_from_reference")
        
        stars = []
        distance_dict = {}
        offsets_x = []
        offsets_y = []
        matches = {}
        
        for refpos in reference_xy:
            Log.info(f"Calculate distances for star at {refpos[0]}, {refpos[1]}")
            
            radius = 300
            
            crop = CropRegion(
                left = round(refpos[0] -radius),
                top = round(refpos[1] -radius),
                width = 2 * radius,
                height = 2 * radius,
                offset_x = 0,
                offset_y = 0)
            
            stars_for_ref = self.build_catalog(crop) 
            if(len(stars_for_ref) > 0):
                distance_dict[refpos[0], refpos[1]] = {}
                for star in stars_for_ref:
                    distance = (
                        refpos[0] - star.xpos,
                        refpos[1] - star.ypos
                    )
                    offsets_x.append(distance[0])
                    offsets_y.append(distance[1])
                    
                    distance_dict[(refpos[0], refpos[1])][(star.xpos, star.ypos)] = distance
                    Log.info(f"{star.xpos},{star.ypos}:{distance[0]},{distance[1]}")
        
        x_min, x_max, y_min, y_max, shift_x, shift_y, best_count = self.calculate_average_shift(
            offsets_x, offsets_y)
        
        for key, innerdict in distance_dict.items():
            
            if(best_count == 1):
                matches[key] = None
                continue    
            
            closest_x = 5000.0
            closest_y = 5000.0
            best_match = None
            for innerkey, distance in innerdict.items():
                if(not(x_min <= distance[0] <= x_max) or not (y_min <= distance[1] <= y_max)):
                    continue
                distance_x = abs(distance[0] - shift_x)
                distance_y = abs(distance[1] - shift_y)
                if(distance_x < closest_x and distance_y < closest_y):
                    closest_x = distance_x
                    closest_y = distance_y
                    Log.info(f"innerkey = {innerkey}")
                    best_match = innerkey
            matches[key] = best_match
            
        Log.info("Matches")
        for key, value in matches.items():
            Log.info(f"{key}:{value}")
        
        return matches
        
        
    def calculate_average_shift(self, offsets_x, offsets_y):
        
        # 1. Arrays intern für die mathematischen Operationen absichern
        arr_x = np.array(offsets_x)
        arr_y = np.array(offsets_y)

        # 2. Suchfenster-Größe festlegen
        max_distance = 20.0 

        best_count = 0
        best_indices = []

        # 3. Gleitendes Fenster berechnen
        for i in range(len(arr_x)):
            center_x = arr_x[i]
            center_y = arr_y[i]
            
            in_box_x = np.abs(arr_x - center_x) <= max_distance / 2
            in_box_y = np.abs(arr_y - center_y) <= max_distance / 2
            hits = in_box_x & in_box_y
            count = np.sum(hits)
            
            if count > best_count:
                best_count = count
                # Speichert die reinen Integer-Indizes ab
                best_indices = np.where(hits)[0].tolist()

        # 4. Werte fehlerfrei über native Python-Listenabstraktion extrahieren
        dense_x = [offsets_x[idx] for idx in best_indices]
        dense_y = [offsets_y[idx] for idx in best_indices]

        # 5. Extremwerte des Wertebereichs bestimmen
        x_min, x_max = min(dense_x), max(dense_x)
        y_min, y_max = min(dense_y), max(dense_y)

        # 6. Verschiebungsmittelwert für Folgeberechnungen ermitteln
        shift_x = sum(dense_x) / len(dense_x)
        shift_y = sum(dense_y) / len(dense_y)

        Log.info(f"Häufigster Wertebereich gefunden!")
        Log.info(f"-> X-Bereich: von {x_min:.4f} bis {x_max:.4f}")
        Log.info(f"-> Y-Bereich: von {y_min:.4f} bis {y_max:.4f}")
        Log.info(f"-> Anzahl der Punkte im Cluster: {best_count} von {len(offsets_x)}")
        Log.info(f"\nBerechneter Verschiebungskern:")
        Log.info(f"-> Delta X = {shift_x:.4f}")
        Log.info(f"-> Delta Y = {shift_y:.4f}")
        
        return x_min, x_max, y_min, y_max, shift_x, shift_y, best_count

        
    
    def make_shape(self, x, y, width, height):

        w = min(self.TILE_SIZE, width - x)
        h = min(self.TILE_SIZE, height - y)

        return [x, y, w, h]
        
        
    def find_star(self, shape):

        tracked_star = None

        try:
            star = self.siril.get_selection_star(shape)
        except Exception as e:
            Log.info(f"No star found in shape {shape}: {e}")
            return None

        if star is None:
            Log.info(f"No star found in shape {shape}")
            return None

        valid, reason = TrackedStar.is_valid(star) 
        if not valid:
            Log.info(f"No star found in shape {shape}: Reason = {reason}")
            return None
            
        Log.info(f"Star found in shape {shape}: {TrackedStar.to_log_string(star)}")

        return star
        
        
    def remove_duplicates(self, stars):

        unique = []

        for star in stars:

            duplicate = False

            for other in unique:

                d = math.hypot(
                    star.xpos - other.xpos,
                    star.ypos - other.ypos
                )

                if d < 8:

                    duplicate = True

                    break

            if not duplicate:

                unique.append(star)

        return unique
 
        
@dataclass
class CropRegion:
    """
    Defines the crop rectangle relative to the tracked star.
    """

    left: float
    top: float

    width: float
    height: float

    offset_x: float
    offset_y: float

    @classmethod
    def from_selection(cls, selection, star_pos):
        """
        Create a CropRegion from the current Siril selection.
        """

        try:
            left, top, width, height = selection

        except TypeError:
            left = selection.x
            top = selection.y
            width = selection.width
            height = selection.height

        return cls(
            left=left,
            top=top,
            width=width,
            height=height,
            offset_x=star_pos[0] - left,
            offset_y=star_pos[1] - top,
        )
        

    def crop_rectangle(self, x, y):
        """
        Calculate the crop rectangle for the current star position.

        Returns:
            (left, top, width, height)
        """

        left = round(x - self.offset_x)
        top = round(y - self.offset_y)

        width = round(self.width)
        height = round(self.height)

        return left, top, width, height


    def crop_command(self, x, y):
        """
        Create the Siril crop command.
        """

        left, top, width, height = self.crop_rectangle(x, y)

        return f"crop {left} {top} {width} {height}"
        
        
def main():

    Log.header(f"StarCrop {Config.VERSION}")

    #
    # Connect to Siril
    #

    try:
        siril = SirilInterface()
        siril.connect()
        Log.info("Connected to Siril.")

    except SirilConnectionError as ex:
        Log.error(ex)
        return

    #
    # Sequence
    #

    try:
        sequence = Sequence(siril)
    except RuntimeError as ex:
        Log.error(ex)
        return

    #
    # Reference star
    #

    reference_stars = siril.get_image_stars()

    if len(reference_stars) == 0:
        Log.error(
            f"Please select at least 1 reference star (5-10 stars recommended)"
            f"(found {len(reference_stars)})."
        )

        return


    #
    # Crop region
    #

    selection = siril.get_siril_selection()

    if selection is None:
        Log.error(
            "No crop region selected."
        )

        return

    crop = CropRegion.from_selection(
        selection,
        (reference_stars[0].xpos, reference_stars[0].ypos) 
    )

    #
    # Information
    #

    Log.header("Sequence")

    Log.info(f"Name           : {sequence.name}")
    Log.info(f"Images         : {sequence.frame_count()}")
    Log.info(f"Current frame  : {sequence.current}")

    Log.header("Crop")

    Log.info(
        f"Size           : "
        f"{crop.width} x {crop.height}"
    )

    Log.info(
        f"Offset         : "
        f"({crop.offset_x:.2f}, "
        f"{crop.offset_y:.2f})"
    )

    Log.info()

    Log.info("Initialization finished.")
    
    
    #
    # Tracker
    #

    tracker = Tracker(
        siril=siril,
        sequence=sequence,
        selection=selection,
    )
    
    #
    # Crop reference frame
    #

    Log.header("Reference frame")

    #
    # Load reference frame
    #

    sequence.load(sequence.current)

    Log.info(
        f"Loaded reference frame {sequence.current+1}"
    )
    
    #reference_catalog = StarCatalogBuilder(siril).build_catalog(crop)
    reference_xy = np.array(
        [[s.xpos, s.ypos] for s in reference_stars]
    )
    
    success = tracker.crop_current_frame((reference_stars[0].xpos, reference_stars[0].ypos), 
        (reference_stars[0].xpos, reference_stars[0].ypos))

    if(success):
        sequence.save(sequence.current)

    Log.info("Reference frame processed.")

    Log.header("Tracking")

    success = tracker.track_sequence(reference_xy)
    
    
if __name__ == "__main__":

    main()  
