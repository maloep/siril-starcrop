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
import math

import sirilpy as s
import sirilpy.enums as senums

s.ensure_installed("astroalign", "numpy", "scikit-image")

from skimage.transform import warp, SimilarityTransform, AffineTransform
import astroalign
import numpy as np
import skimage


siril = SirilInterface()


class Config:
    """
    Global configuration.
    """

    VERSION = "0.1.0-alpha1"

    CROP = True
    ALIGN = True

    TRACE = 10
    DEBUG = 20
    INFO = 30
    WARNING = 40
    ERROR = 50
    NONE = 100

    current_level = TRACE    

    MAX_MAGNITUDE_DELTA = 0.5
    MAX_MAGNITUDE = 4.8

    MIN_SNR = 8.0

    MAX_FWHMX = 10.0
    MAX_FWHMY = 10.0


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

    
    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------

    OUTPUT_PREFIX = "starcrop_"
    

class Log:

    

    @classmethod
    def set_level(cls, level: int):        
        Config.current_level = level

    @classmethod
    def _should_log(cls, level: int) -> bool:
        return level >= Config.current_level

    @classmethod
    def trace(cls, message: str):
        if cls._should_log(Config.TRACE):
            siril.log(f"[TRACE] {message}")

    @classmethod
    def debug(cls, message: str):
        if cls._should_log(Config.DEBUG):
            siril.log(f"[DEBUG] {message}")

    @classmethod
    def info(cls, message: str):
        if cls._should_log(Config.INFO):
            siril.log(f"[INFO] {message}")

    @classmethod
    def warning(cls, message: str):
        if cls._should_log(Config.WARNING):
            siril.log(f"[WARNING] {message}", senums.LogColor.SALMON)

    @classmethod
    def error(cls, message: str):
        if cls._should_log(Config.ERROR):
            siril.log(f"[ERROR] {message}", senums.LogColor.RED)

    @classmethod
    def success(cls, message: str):        
        siril.log(f"[SUCCESS] {message}", senums.LogColor.GREEN)
    
    @classmethod
    def header(cls, text):
        siril.log("\n")
        siril.log("=" * 60)
        siril.log(text)
        siril.log("=" * 60)
        

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


    def __init__(self, ):        
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
                siril.cmd(
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

        siril.cmd(
            f"load {self.name}{frame_index + 1:05d}{self.extension}"
        )
        
        self.current = frame_index

    def first_frame(self):
        return self.seq.beg - 1


    def last_frame(self):
        return self.seq.end - 1


    def frame_count(self):
        return self.number
        
    def get_filename(self, frame_index):
        return (
            f"{Config.OUTPUT_PREFIX}"
            f"{frame_index + 1:05d}"
            f"{self.extension}"
        )
        
    
    def save(self, frame_index):
        """
        Save the currently loaded image.
        """
        filename = self.get_filename(frame_index)

        Log.info(f"Saving {filename}")

        siril.cmd(f"save {filename}")
        

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
    def is_valid(cls, star, strict):
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

        if (not strict):
            Log.debug("is_valid() strict == false. Won't check star parameters.")
            return True, ""

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
        sequence,
        selection
    ):

        self.sequence = sequence
        self.selection = selection
        self.transform_dict = {}


    def track_sequence(self, reference_positions, cropped_reference_positions, refcropleft, refcroptop, ref_pixeldata):
        """
        Track the reference star through the whole sequence.
        """

        #Reference image has no transformation matrix
        self.transform_dict[1] = None

        while self.track_next_frame(reference_positions, cropped_reference_positions, refcropleft, refcroptop, ref_pixeldata):
            pass
            
        if(Config.ALIGN):
            self.create_complete_siril_seq()

        Log.header("Tracking finished")
    
    
    def track_next_frame(self, reference_positions, cropped_reference_positions, refcropleft, refcroptop, ref_pixeldata):
        #
        # Next frame
        #

        next_frame = self.sequence.current + 1

        if(next_frame == 5):
            Log.info(f"User defined stop at frame {next_frame}.")
            return False

        if next_frame >= self.sequence.frame_count():
            Log.info("Last frame reached.")
            return False

        Log.header(f"Frame {next_frame +1}")

        #
        # Load frame
        #

        self.sequence.load(next_frame)        
        Log.info(f"Loaded frame {next_frame +1}")

        #
        # Search
        #

        ref_xy, match_xy = self.find_matching_stars_from_catalog(reference_positions)

        if ref_xy is None or match_xy is None:
            Log.warning("no reference star found for current frame.")
            return True
        else:            
            Log.success(
                f"Found matching star: "
                f"({ref_xy[0][0]:.2f}, "
                f"{ref_xy[0][1]:.2f})"
                f"({match_xy[0][0]:.2f}, "
                f"{match_xy[0][1]:.2f})"
            )
        
        if(Config.CROP):
            #use 1st match
            success, matchcropleft, matchcroptop = self.crop_current_frame(ref_xy[0], match_xy[0])
            
            if(success):
                self.sequence.save(next_frame)

            cropped_ref_xy = []
            cropped_match_xy = []
                
            for refpos in ref_xy:
                cropped_refpos = (refpos[0]-refcropleft, refpos[1]-refcroptop)
                cropped_ref_xy.append(self.translate_calculated_refpos_in_real_refpos(cropped_refpos, cropped_reference_positions))

            for matchpos in match_xy:
                cropped_match_xy.append((matchpos[0]-matchcropleft, matchpos[1]-matchcroptop))
                        
            Log.trace(f"cropped_match_xy = {cropped_match_xy}")
            
            Log.debug("recalculate matched stars from cropped image")
            sBuilder = StarCatalogBuilder(siril) 
            current_matches = sBuilder.build_matches_from_reference(cropped_match_xy, 50, 100, 100, False)
            cropped_current_positions = []
                               
            for refpos, matchpos in current_matches.items():
                if(matchpos is None):
                    #current_positions.append(None)
                    continue                 
                cropped_current_positions.append((matchpos[0], matchpos[1]))
            Log.trace(f"current_positions (recalculated) = {cropped_current_positions}")
                
            
        if(Config.ALIGN and (not Config.CROP or success)):                        

            if(Config.CROP):
                self.align(cropped_ref_xy, cropped_current_positions, ref_pixeldata, next_frame)
            else:
                self.align(ref_xy, match_xy, ref_pixeldata, next_frame)

        return True
        
        
    def find_matching_stars_from_catalog(self, reference_positions):
        
        sBuilder = StarCatalogBuilder(siril)
        
        matches = sBuilder.build_matches_from_reference(reference_positions, 300, 100, 75, True)
        
        ref_xy = []
        match_xy = []
        
        for refpos, matchpos in matches.items():
            if(matchpos is not None):
                ref_xy.append((refpos[0], refpos[1]))
                match_xy.append((matchpos[0], matchpos[1]))
        
        Log.trace(f"ref_xy = {ref_xy}")
        Log.trace(f"match_xy = {match_xy}")
        
        if(len(ref_xy) == 0 or len(match_xy) == 0):
            Log.warning("No matches found for current frame")
            return None, None
                
        return ref_xy, match_xy
    

    def translate_calculated_refpos_in_real_refpos(self, cropped_refpos, cropped_reference_positions):

        cropped_refpos = np.array(cropped_refpos)

        distances = np.sqrt(np.sum((cropped_reference_positions - cropped_refpos)**2, axis=1))

        closest_index = np.argmin(distances)

        return cropped_reference_positions[closest_index]

              
    def align(self, cropped_ref_xy, cropped_match_xy, ref_pixeldata, next_frame):

        #transform = AffineTransform()
        #transform.estimate(cropped_ref_xy, cropped_match_xy)

        transform = astroalign.estimate_transform(
            'affine',
            cropped_ref_xy,
            cropped_match_xy
        )
        Log.debug(f"estimate_transform {transform}")

        cropped_ref_xy = np.asarray(cropped_ref_xy, dtype=np.float64)
        cropped_match_xy = np.asarray(cropped_match_xy, dtype=np.float64)

        # 1. Calculate transformation + map individual pixel errors
        try:
            # Map 'current' stars to where they SHOULD be on the reference frame
            predicted_xy = transform(cropped_match_xy)
            
            # Calculate the exact Euclidean pixel distance error for every star pair
            pixel_errors = np.linalg.norm(predicted_xy - cropped_ref_xy, axis=1)
            
            print("--- INDIVIDUAL PIXEL ERRORS ---")
            for idx, error in enumerate(pixel_errors):
                print(f"Star Pair {idx + 1}: Error = {error:.2f} pixels")
                
        except Exception as e:
            print(f"Initial mapping failed: {e}")


        # 2. AUTOMATED CHECK: Use RANSAC to isolate bad matches 
        from skimage.measure import ransac
        from skimage.transform import AffineTransform

        # RANSAC will test combinations and find the consensus
        # min_samples=3 is the minimum required points to define an Affine plane
        # residual_threshold=3.0 means any star misaligned by > 3 pixels is flagged as an outlier
        model_robust, inliers = ransac(
            (cropped_match_xy, cropped_ref_xy), 
            AffineTransform, 
            min_samples=3, 
            residual_threshold=3.0, 
            max_trials=100
        )

        print("\n--- AUTOMATED QUALITY CHECK ---")
        print(f"Total stars analyzed: {len(cropped_match_xy)}")
        print(f"Good matches (Inliers): {np.sum(inliers)}")
        print(f"Bad matches (Outliers): {np.sum(~inliers)}")

        # Isolate the clean star list programmatically
        clean_current = cropped_match_xy[inliers]
        clean_reference = cropped_ref_xy[inliers]

        # Identify exactly which indices failed
        outlier_indices = np.where(~inliers)[0]
        if len(outlier_indices) > 0:
            print(f"⚠️ Action required: Star indices {outlier_indices + 1} failed the quality threshold.")

            
        for idx in range(len(cropped_ref_xy)):
            point = cropped_ref_xy[idx]
            result_point = transform(point)

            Log.debug(f"Quellstern bei: {point}")
            Log.debug(f"Soll landen bei: {cropped_match_xy[idx]}")
            Log.debug(f"Landet real bei: {result_point}")  
        
        if(len(model_robust.params) < 3):
            Log.warning("Less than 3 matching pair of stars fount for current image.")
            self.transform_dict[next_frame +1] = None
        else:
            self.transform_dict[next_frame +1] = model_robust
        
            
        return
        
        with siril.image_lock():
            #fit_obj = siril.get_image()
            Log.info(f"Image filename = {siril.get_image_filename()}")
            target_data = siril.get_image_pixeldata()
            
            print("Target Shape:", target_data.shape)  # MUSS bei RGB (3, Höhe, Breite) sein, NICHT (Höhe, Breite, 3)
            print("Target Dtype:", target_data.dtype)  # MUSS float32 sein, NICHT float64
            print("Target Min/Max:", target_data.min(), target_data.max())  # Sollte exakt zwischen 0.0 und 1.0 liegen
            
            print("Ref Shape:", ref_pixeldata.shape)  # MUSS bei RGB (3, Höhe, Breite) sein, NICHT (Höhe, Breite, 3)
            print("Ref Dtype:", ref_pixeldata.dtype)  # MUSS float32 sein, NICHT float64
            print("Ref Min/Max:", ref_pixeldata.min(), ref_pixeldata.max())  # Sollte exakt zwischen 0.0 und 1.0 liegen
            
            channel_r = target_data[0, :, :]
            channel_g = target_data[1, :, :]
            channel_b = target_data[2, :, :]
            #registered_data, footprint = astroalign.apply_transform(transform, target_data, ref_pixeldata)
            
            warped_r = warp(channel_r, transform.inverse, cval=0.0)
            warped_g = warp(channel_g, transform.inverse, cval=0.0)
            warped_b = warp(channel_b, transform.inverse, cval=0.0)
            
            registered_data = np.stack([warped_r, warped_g, warped_b], axis=0)
            
            #registered_data = np.nan_to_num(registered_data, nan=0.0, posinf=0.0, neginf=0.0)
            registered_data = registered_data.astype(np.float32)
            registered_data = np.clip(registered_data, 0.0, None)
            
            print("Mittelwert des Bildes:", np.mean(registered_data))
            print("Shape:", registered_data.shape)  # MUSS bei RGB (3, Höhe, Breite) sein, NICHT (Höhe, Breite, 3)
            print("Dtype:", registered_data.dtype)  # MUSS float32 sein, NICHT float64
            print("Min/Max:", registered_data.min(), registered_data.max())  # Sollte exakt zwischen 0.0 und 1.0 liegen
            
            siril.set_image_pixeldata(registered_data)
            siril.cmd("save", f"r_cropped_{next_frame:05d}.fit")


    def create_complete_siril_seq(self):
        """
        Creates a complete .seq file from scratch (compatible with Siril 1.4.x).
        """        

        seq_filepath = f"{Config.OUTPUT_PREFIX}.seq"
        start_idx = 1
        num_images = len(self.transform_dict.keys())
        fixed_len = 5       #5 for 00001
        ref_image_idx = 1   #index of reference image

        Log.info(f"Write sequence file: {seq_filepath}")
        
        # Berechne den relativen Referenz-Index für den Header (Siril intern oft 0-basiert ab Start)
        rel_ref_idx = ref_image_idx - start_idx
        
        with open(seq_filepath, 'w', encoding='utf-8') as f:
            # 1. Standard-Kommentare schreiben
            f.write("#Siril sequence file. Contains list of images, selection, registration data and statistics\n")
            f.write("#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version variable_size fz_flag drizzle\n")
            
            # 2. Header-Zeile generieren (nb_selected wird auf num_images gesetzt, Version = 6)
            f.write(f"S '{Config.OUTPUT_PREFIX}' {start_idx} {num_images} {num_images} {fixed_len} {rel_ref_idx} 6 0 0 0\n")
            
            # 3. Kanal-Indikator (L 3 steht für vordefinierte Layer-Eigenschaften, Standardwert)
            f.write("L 3\n")
            
            for img_num in self.transform_dict.keys():
                # 'I <Bildnummer> 1' deklariert das Bild als aktiv/ausgewählt
                f.write(f"I {img_num} 1\n")

            # 4. Bilder und Registrierungen schreiben
            for img_num in self.transform_dict.keys():
                # Prüfen, ob für dieses Bild ein Astroalign-Ergebnis existiert
                if img_num in self.transform_dict and self.transform_dict[img_num] is not None:
                    matrix = self.transform_dict[img_num].params
                    
                    # Falls Astroalign eine affine 2x3 Matrix wirft, auf homogene 3x3 erweitern
                    if matrix.shape == (2, 3):
                        T = np.vstack([matrix, [0, 0, 1]])
                    else:
                        T = matrix
                    
                    # Verschiebungswerte extrahieren
                    dx = T[0, 2]
                    dy = T[1, 2]
                    
                    # Matrix flachklopfen für die Ausgabe
                    m = T.flatten()
                    
                    # R1 Zeile (Kanal 1) mit Dummy-Sternanzahl (100) und FWHM schreiben
                    r1_line = f"R1 {dx:.4f} {dy:.4f} 0.000000 0 0.000000 100 H " \
                            f"{m[0]:.6g} {m[1]:.6g} {m[2]:.6g} " \
                            f"{m[3]:.6g} {m[4]:.6g} {m[5]:.6g} " \
                            f"{m[6]:.6g} {m[7]:.6g} {m[8]:.6g}\n"
                    f.write(r1_line)
                else:
                    # Referenzbild oder Bilder ohne Transformation (Einheitsmatrix)
                    # Bei der Referenz selbst ist dx=0.0 und dy=0.0
                    f.write("R1 0 0 0 0 0 0 H 1 0 0 0 1 0 0 0 1\n")

        Log.success(f"Sequence file successfully written: {seq_filepath}")


    def crop_current_frame(self, source_pos, target_pos):
        """
        Crop the currently loaded image.
        """
        crop = CropRegion.from_selection(
            self.selection,
            source_pos
        )

        command, left, top = crop.crop_command(target_pos[0], target_pos[1])

        try:
            siril.cmd(command)
        except Exception as e:
            Log.warning(f"Error while cropping frame: {e.message}")
            return False, 0.0, 0.0

        Log.success(f"Image cropped: {command}")
        return True, left, top
 
 
class StarCatalogBuilder:

    def __init__(self, siril):
        siril = siril


    def build_catalog(self, crop, tile_size, tile_step, strict, reference_star=None):
        
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
                    height +crop.top,
                    tile_size
                )

                star = self.find_star(shape, strict)

                if star is not None:
                    stars.append(star)

                x += tile_step

            y += tile_step

        stars = self.remove_duplicates(stars)

        Log.trace(f"Catalog {crop} contains {len(stars)} stars.")

        return stars
        
        
    def build_matches_from_reference(self, reference_xy, radius, tile_size, tile_step, strict):
                
        distance_dict = {}
        offsets_x = []
        offsets_y = []
        matches = {}
        
        for refpos in reference_xy:
            Log.trace(f"Calculate distances for star at {refpos[0]}, {refpos[1]}")
            
            search = CropRegion(
                left = round(refpos[0] -radius),
                top = round(refpos[1] -radius),
                width = 2 * radius,
                height = 2 * radius,
                offset_x = 0,
                offset_y = 0)
            
            stars_for_ref = self.build_catalog(search, tile_size, tile_step, strict) 
            distance_dict[refpos[0], refpos[1]] = {}
            if(len(stars_for_ref) > 0):
                for star in stars_for_ref:
                    distance = (
                        refpos[0] - star.xpos,
                        refpos[1] - star.ypos
                    )
                    offsets_x.append(distance[0])
                    offsets_y.append(distance[1])
                    
                    distance_dict[(refpos[0], refpos[1])][(star.xpos, star.ypos)] = distance
                    Log.trace(f"{star.xpos},{star.ypos}:{distance[0]},{distance[1]}")
        
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
                    Log.trace(f"innerkey = {innerkey}")
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

        Log.debug(f"Häufigster Wertebereich gefunden!")
        Log.debug(f"-> X-Bereich: von {x_min:.4f} bis {x_max:.4f}")
        Log.debug(f"-> Y-Bereich: von {y_min:.4f} bis {y_max:.4f}")
        Log.debug(f"-> Anzahl der Punkte im Cluster: {best_count} von {len(offsets_x)}")
        Log.debug(f"\nBerechneter Verschiebungskern:")
        Log.debug(f"-> Delta X = {shift_x:.4f}")
        Log.debug(f"-> Delta Y = {shift_y:.4f}")
        
        return x_min, x_max, y_min, y_max, shift_x, shift_y, best_count
        
    
    def make_shape(self, x, y, width, height, tile_size):

        w = min(tile_size, width - x)
        h = min(tile_size, height - y)

        return [x, y, w, h]
        
        
    def find_star(self, shape, strict):

        try:
            star = siril.get_selection_star(shape)
        except Exception as e:
            Log.trace(f"No star found in shape {shape}: {e}")
            return None

        if star is None:
            Log.trace(f"No star found in shape {shape}")
            return None

        valid, reason = TrackedStar.is_valid(star, strict)
        if not valid:
            Log.trace(f"No star found in shape {shape}: Reason = {reason}")
            return None
            
        Log.trace(f"Star found in shape {shape}: {TrackedStar.to_log_string(star)}")

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

        return f"crop {left} {top} {width} {height}", left, top
        
        
def main():

    #
    # Connect to Siril
    #

    try:        
        siril.connect()

        Log.header(f"StarCrop {Config.VERSION}")        
        Log.info("Connected to Siril.")

    except SirilConnectionError as ex:
        print(ex)
        return

    #
    # Sequence
    #

    try:
        sequence = Sequence()
    except RuntimeError as ex:
        Log.error(ex)
        return

    #
    # Reference star
    #

    try:
        reference_stars = siril.get_image_stars()
    except Exception as ex:
        Log.error("Error detecting selected stars. Please select at least 1 reference star (5-10 stars recommended)")
        return

    if len(reference_stars) == 0:
        Log.error(
            f"Please select at least 1 reference star (5-10 stars recommended)"
            f"(found {len(reference_stars)})."
        )

        return
        
    reference_positions = np.array(
        [[s.xpos, s.ypos] for s in reference_stars]
    )

    #
    # Crop region
    #

    if(Config.CROP):
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
    else:
        [channels, width, height] = siril.get_image_shape()
        crop = CropRegion(0.0, 0.0, width, height, 0.0, 0.0)

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

    Log.info("Initialization finished.")
    
    
    #
    # Tracker
    #

    tracker = Tracker(
        sequence=sequence,
        selection=selection
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

    cropped_reference_positions = []
    cropleft = 0.0
    croptop = 0.0
    
    if(Config.CROP):
        success, cropleft, croptop = tracker.crop_current_frame((reference_stars[0].xpos, reference_stars[0].ypos), 
            (reference_stars[0].xpos, reference_stars[0].ypos))

        if(success):
            sequence.save(sequence.current)

            cropped_ref_positions_tmp = []
            for refpos in reference_positions:
                cropped_ref_positions_tmp.append((refpos[0]-cropleft, refpos[1]-croptop))

            Log.trace(f"calculated cropped_ref_positions = {cropped_ref_positions_tmp}")

            Log.debug("rematch reference stars from cropped image")
            
            sBuilder = StarCatalogBuilder(siril)        
            refmatches = sBuilder.build_matches_from_reference(cropped_ref_positions_tmp, 50, 100, 100, False)
            for refpos, matchpos in refmatches.items():
                if(matchpos is None):
                    cropped_reference_positions.append(None)
                    continue
                cropped_reference_positions.append((matchpos[0], matchpos[1]))
            Log.trace(f"reference_positions (rematched) = {cropped_reference_positions}")
    else:
        left = selection.x
        top = selection.y
        
    Log.info(f"Image filename = {siril.get_image_filename()}")
        
    reference_pixeldata = siril.get_image_pixeldata()

    Log.info("Reference frame processed.")

    Log.header("Tracking")

    success = tracker.track_sequence(reference_positions, cropped_reference_positions, cropleft, croptop, reference_pixeldata)
    
    
if __name__ == "__main__":

    main()  
