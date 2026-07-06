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

    MIN_SNR = 5.0

    MAX_FWHM_DELTA = 5.0

    
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
    Represents the reference star and its current position.
    """

    reference_x: float
    reference_y: float

    current_x: float
    current_y: float

    magnitude: float

    snr: float

    fwhmx: float
    fwhmy: float
    
    @property
    def current_position(self):
        return (self.current_x, self.current_y)

    @property
    def reference_position(self):
        return (self.reference_x, self.reference_y)

    @classmethod
    def from_psf_star(cls, star):
        """
        Create a TrackedStar from a Siril PSFStar object.
        """

        return cls(
            reference_x=star.xpos,
            reference_y=star.ypos,

            current_x=star.xpos,
            current_y=star.ypos,

            magnitude=star.mag,

            snr=star.SNR,

            fwhmx=star.fwhmx,
            fwhmy=star.fwhmy,
        )

    def update(self, star):
        """
        Update the current position after a successful match.
        """

        self.current_x = star.xpos
        self.current_y = star.ypos
        
        self.last_star = star

    def magnitude_difference(self, star):

        return abs(self.magnitude - star.mag)

    def fwhm_difference(self, star):

        return max(
            abs(self.fwhmx - star.fwhmx),
            abs(self.fwhmy - star.fwhmy),
        )

    
    def is_valid(self, star):
        """
        Validate a detected star.
        """

        import math

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

        delta_mag = abs(self.magnitude - star.mag)

        if delta_mag > Config.MAX_MAGNITUDE_DELTA:

            return (
                False,
                f"ΔMag={delta_mag:.2f} "
                f"(>{Config.MAX_MAGNITUDE_DELTA})",
            )

        if star.SNR < Config.MIN_SNR:

            return (
                False,
                f"SNR={star.SNR:.2f} "
                f"(<{Config.MIN_SNR})",
            )

        delta_fwhm = max(
            abs(self.fwhmx - star.fwhmx),
            abs(self.fwhmy - star.fwhmy),
        )

        if delta_fwhm > Config.MAX_FWHM_DELTA:

            return (
                False,
                f"ΔFWHM={delta_fwhm:.2f} "
                f"(>{Config.MAX_FWHM_DELTA})",
            )

        return True, ""
        


class Tracker:

    def __init__(
        self,
        siril,
        sequence,
        tracked_star,
        crop_region,
    ):

        self.siril = siril
        self.sequence = sequence
        self.reference_star = tracked_star
        self.crop = crop_region


    def track_sequence(self):
        """
        Track the reference star through the whole sequence.
        """

        while self.track_next_frame():
            pass

        Log.header("Tracking finished")
    
    
    def track_next_frame(self):
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

        found_star = self.find_star()

        if found_star is None:
            Log.warning("Reference star not found.")
            return True
        else:            
            Log.info(
                f"Found star: "
                f"({found_star.xpos:.2f}, "
                f"{found_star.ypos:.2f})"
            )

            Log.info(
                f"Mag={found_star.mag:.2f} "
                f"SNR={found_star.SNR:.2f}"
            )

        #
        # Update
        #

        self.reference_star.update(found_star)

        Log.info(
            f"Star found at "
            f"({found_star.xpos:.2f}, "
            f"{found_star.ypos:.2f})"
        )
        
        self.crop_current_frame()        
        self.sequence.save(next_frame)

        return True
        
    
    def search_at(self, x, y, radius, label):
        """
        Search for the reference star around a given position.
        """
        import math

        left = round(x - radius)
        top = round(y - radius)

        width = radius * 2
        height = radius * 2
        
        #
        # Siril rejects selections larger than about 300x300.
        #
        
        width = min(width, 300)
        height = min(height, 300)

        #
        # Keep search window inside the image.
        #

        if left < 0:

            width += left
            left = 0

        if top < 0:

            height += top
            top = 0

        if left + width > self.sequence.width:
            width = self.sequence.width - left

        if top + height > self.sequence.height:
            height = self.sequence.height - top

        if width <= 0 or height <= 0:
            Log.info("Search window outside image.")

            return None

        shape = [left, top, width, height]

        Log.info("")
        Log.info(f"Searching {label}")
        Log.info(f"Radius : {radius}")
        Log.info(f"Shape  : {shape}")

        try:
            star = self.siril.get_selection_star(shape)
        except Exception as e:
            Log.warning(f"Search failed: {e}")
            return None

        if star is None:
            Log.info("No candidate found.")
            return None

        Log.info(
            f"Candidate: "
            f"({star.xpos:.2f}, {star.ypos:.2f})"
        )

        Log.info(
            f"Mag={star.mag:.2f} "
            f"SNR={star.SNR:.2f} "
            f"FWHM={star.fwhmx:.2f}/{star.fwhmy:.2f}"
        )

        valid, reason = self.reference_star.is_valid(star)

        if not valid:
            Log.info(f"Rejected: {reason}")
            return None

        Log.info("Candidate accepted.")

        return star
        
        
    def search_grid(self, center_x, center_y, label):
        """
        Search a 3x3 grid using a fixed 300x300 search window.
        """

        candidates = []

        radius = 150

        for dx, dy in Config.GRID_OFFSETS:
            star = self.search_at(
                center_x + dx,
                center_y + dy,
                radius,
                f"{label}-grid ({dx:+d},{dy:+d})",
            )

            if star is not None:
                candidates.append(star)

        return candidates

    
    def find_star(self):
        """
        Find the reference star.
        """

        candidates = []

        #
        # Local search
        #

        for radius in Config.SEARCH_RADII:
            star = self.search_at(
                self.reference_star.current_x,
                self.reference_star.current_y,
                radius,
                "current",
            )

            if star is not None:
                candidates.append(star)

            star = self.search_at(
                self.reference_star.reference_x,
                self.reference_star.reference_y,
                radius,
                "reference",
            )

            if star is not None:
                candidates.append(star)

        #
        # Grid search around current position
        #

        Log.info("")
        Log.info("Starting grid search around current position.")

        candidates.extend(self.search_grid(
            self.reference_star.current_x,
            self.reference_star.current_y,
            "current",
        ))

       
        #
        # Grid search around reference position
        #

        Log.info("")
        Log.info("Starting grid search around reference position.")

        candidates.extend(self.search_grid(
            self.reference_star.reference_x,
            self.reference_star.reference_y,
            "reference",
        ))
        
        if len (candidates) > 0:
            star = min(candidates, key=self.score)
        
        if star is None:
            Log.warning("Reference star not found")
            return None

        return star
        
    
    def score(self, star):  
        import math

        delta_mag = abs(
            star.mag - self.reference_star.magnitude
        )

        delta_fwhm = max(
            abs(star.fwhmx - self.reference_star.fwhmx),
            abs(star.fwhmy - self.reference_star.fwhmy),
        )

        distance = math.hypot(
            star.xpos - self.reference_star.reference_x,
            star.ypos - self.reference_star.reference_y,
        )
        
        Log.info(
            f"star={star.xpos:.2f}, {star.ypos:.2f}, "
            f"Delta_Mag={delta_mag:.2f}, "
            f"delta_fwhm={delta_fwhm:.2f}, "
            f"distance={distance:.2f}"
        )

        return (
            distance,
        )
        
        
    def crop_current_frame(self):
        """
        Crop the currently loaded image.
        """

        command = self.crop.crop_command(self.reference_star)

        Log.info(command)

        self.siril.cmd(command)

        Log.info("Image cropped.")
  
        
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
    def from_selection(cls, selection, star):
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
            offset_x=star.xpos - left,
            offset_y=star.ypos - top,
        )

    def crop_position(self, tracked_star):
        """
        Calculate the crop position for the current star position.

        Returns:
            left, top
        """

        left = tracked_star.current_x - self.offset_x
        top = tracked_star.current_y - self.offset_y

        return round(left), round(top)
        

    def crop_rectangle(self, tracked_star):
        """
        Calculate the crop rectangle for the current star position.

        Returns:
            (left, top, width, height)
        """

        left = round(tracked_star.current_x - self.offset_x)
        top = round(tracked_star.current_y - self.offset_y)

        width = round(self.width)
        height = round(self.height)

        return left, top, width, height


    def crop_command(self, tracked_star):
        """
        Create the Siril crop command.
        """

        left, top, width, height = self.crop_rectangle(tracked_star)

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

    stars = siril.get_image_stars()

    if len(stars) != 1:
        Log.error(
            f"Exactly one reference star must be selected "
            f"(found {len(stars)})."
        )

        return

    reference_star = TrackedStar.from_psf_star(
        stars[0]
    )

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
        stars[0]
    )

    #
    # Information
    #

    Log.header("Sequence")

    Log.info(f"Name           : {sequence.name}")
    Log.info(f"Images         : {sequence.frame_count()}")
    Log.info(f"Current frame  : {sequence.current}")

    Log.header("Reference star")

    Log.info(
        f"Position       : "
        f"({reference_star.reference_x:.2f}, "
        f"{reference_star.reference_y:.2f})"
    )

    Log.info(
        f"Magnitude      : "
        f"{reference_star.magnitude:.2f}"
    )

    Log.info(
        f"SNR            : "
        f"{reference_star.snr:.2f}"
    )

    Log.info(
        f"FWHM           : "
        f"{reference_star.fwhmx:.2f} / "
        f"{reference_star.fwhmy:.2f}"
    )

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
        tracked_star=reference_star,
        crop_region=crop,
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
        f"Loaded reference frame {sequence.current}"
    )    
    
    tracker.crop_current_frame()

    sequence.save(sequence.current)

    Log.info("Reference frame processed.")

    Log.header("Tracking")

    success = tracker.track_sequence()
    
    
if __name__ == "__main__":

    main()
