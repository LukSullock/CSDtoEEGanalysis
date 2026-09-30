# -*- coding: utf-8 -*-
"""
CSD to EEG conversion. Scripts used to convert CSD to EEG and analyse resulting data.

Copyright (C) 2026 Luk Sullock Enzlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from MacacaDataExtractionFunctions import ExtractData

exitcodemap = {
    1: "User decision"
    }

# ImportFolder = "F:\\transfer_3433643_files_9dcc8245\\V4PoP (Jan 2026 bug fixes)\\C190219-npy"
ImportFolder = "F:\\transfer_3433643_files_9dcc8245\\V4PoP (Jan 2026 bug fixes)"
ExportFolder = "F:\\saveddata\\260930-All"
DataFile = "csd_array_naPmm3.npy" # Name of the datafile to be imported
TimeFile = "time_array_ms.npy" # Name of the time file to be imported
ProbeFile = "probe.csv"
EEGFile = "eeg_array_uv.npy"
DFHeaderSelection = {
    "SessionInfo": { # It's important that SessionInfo with date and monkey_identifier
                     #  are included for the data organisation structure
        "recordinginfo.csv":    ["date",
                                 "investigator",
                                 "monkey_identifier",
                                 "recorded_hemisphere",
                                 "recorded_area",
                                 "probe_manufacturer",
                                 "receptive_field_position_deg",
                                 "data_sampling_rate_hz"]
        },
    "Data": {
        "task.csv":             ["trial_number_count",
                                 "time_in_recording_s",
                                 "catch_trial_logical",
                                 "target_present_logical",
                                 "distractors_present_logical",
                                 "target_color_string",
                                 "distractor_color_string",
                                 "array_set_size_count",
                                 "array_target_position_deg",
                                 "block_count",
                                 "block_length_count",
                                 "block_trial_count"],
        "behavior.npy":         ["reaction_time_ms",
                                 "accuracy_logical",
                                 "trial_outcome_code"]
        },
    "ChannelInfo": {
        "probe.csv":            ["contact",
                                 "layer_category"]
        }
    }
InfoChecks = {
    "SessionInfo": {
        "date": [],
        "investigator": "westerberg",
        "monkey_identifier": {"C": "C", "H": "H"},
        "recorded_hemisphere": {"C": "L", "H": "R"},
        "recorded_area": "V4",
        "probe_manufacturer": "Plexon"
        },
    "Data": {
        "array_set_size_count": 6
        }
    }
DataSelection = {
    "SessionInfo": {
        "monkey_identifier": ["C", "H"]
        },
    "Data": {
        "catch_trial_logical": 0,
        "target_present_logical": 1,
        "distractors_present_logical": 1,
        "array_set_size_count": 6,
        "reaction_time_ms": {">=": 0.0},
        "accuracy_logical": 1,
        "trial_outcome_code": [2, 3, 4, 5, 6, 7, 8, 14]
        }
    }
BaselineTime = 100 # ms
MinDataPoints = 50 # Minimum amount of data points per group
DegConversion = 90 # By default 0-degrees points to the right and 90-degrees to the top.
                        # This value gets added to the degrees. Adding 90 means 0-degrees
                        # points to the top.

DataSeparationConditions = {
    "TargetUnprimed": {
        "Data": {
            "array_target_position_deg": {
                "C": {">": 180, "<": 360, "match": "receptive_field_position_deg"},
                "H": {"<": 180, ">": 0, "match": "receptive_field_position_deg"}
                },
            "block_trial_count": {"<=": 3}
            }
        },
    "TargetPrimed": {
        "Data": {
            "array_target_position_deg": {
                "C": {">": 180, "<": 360, "match": "receptive_field_position_deg"},
                "H": {"<": 180, ">": 0, "match": "receptive_field_position_deg"}
                },
            "block_trial_count": {">=": 4}
            }
        },
    "DistractorUnprimed": {
        "Data": {
            "array_target_position_deg": {
                "C": {"<": 180, ">": 0, "mismatch": "receptive_field_position_deg"},
                "H": {">": 180, "<": 360, "mismatch": "receptive_field_position_deg"}
                },
            "block_trial_count": {"<=": 3}
            }
        },
    "DistractorPrimed": {
        "Data": {
            "array_target_position_deg": {
                "C": {"<": 180, ">": 0, "mismatch": "receptive_field_position_deg"},
                "H": {">": 180, "<": 360, "mismatch": "receptive_field_position_deg"}
                },
            "block_trial_count": {">=": 4}
            }
        }
    }

SessionAverageSelection = [ # Due to the mean being taken, only numerical columns will be
                            #   kept (won't raise an error)
    "reaction_time_ms"
    ]

exitcode, DFs, Arrays = ExtractData(ImportFolder, ExportFolder, DFHeaderSelection, DataFile, TimeFile,
                       InfoChecks, DataSelection, BaselineTime, MinDataPoints, DegConversion,
                       DataSeparationConditions, SessionAverageSelection, ProbeFile, EEGFile)
if exitcode: print(f'\033[91mExitcode: \033[0m{exitcode} ({exitcodemap[exitcode]})')
else: print("\n\033[1;32mData succesfully exported.\033[0m")
