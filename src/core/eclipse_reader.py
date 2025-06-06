"""Eclipse file reader for reservoir simulation data.

This module provides functionality for reading Eclipse simulation files
and extracting relevant data for the PINO model.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import resdata
from resdata import grid, summary, resfile, rft, well
import os
import gc

# Print logging info
print(f"Current directory: {os.getcwd()}")
print(f"Log file will be written to: {os.path.join(os.getcwd(), 'prepare_training_data.log')}")

# Ensure log directory exists and is writable
log_dir = "../../logs"  # Change this to your desired directory
os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'prepare_training_data.log'))
    ]
)

logger = logging.getLogger(__name__)  # Add a logger for this module

def normalize_well_name(well_name: str) -> str:
    """Normalize well names to handle potential inconsistencies between different data sources.
    
    Args:
        well_name: The original well name
        
    Returns:
        Normalized well name
    """
    # Remove any quote marks
    normalized = well_name.strip("'\"")
    
    # Handle common variations (add more patterns as needed)
    # For example, some files may use "WELL-01" while others use "WELL_01" or "WELL01"
    normalized = normalized.replace('-', '').replace('_', '')
    
    return normalized

def parse_welspecs_section(content: str) -> Dict[str, str]:
    """Parse the WELSPECS section from Eclipse SCHEDULE content to extract well types.
    
    Args:
        content: The content of the SCHEDULE file or section containing WELSPECS
        
    Returns:
        Dictionary mapping well names to their types (producer, water_injector, gas_injector)
    """
    well_types = {}
    original_well_names = {}  # Map normalized names to original names
    normalized_to_type = {}   # Temporary storage for normalized names to types
    
    try:
        # Find the WELSPECS section
        if 'WELSPECS' not in content:
            logger.warning("WELSPECS keyword not found in content")
            return well_types
            
        # Extract the WELSPECS section
        welspecs_start = content.find('WELSPECS')
        if welspecs_start == -1:
            logger.warning("Cannot locate WELSPECS keyword in content")
            return well_types
            
        logger.info(f"Found WELSPECS keyword at position {welspecs_start}")
        
        # Find the end of the section (either '/' on a line by itself or another keyword)
        content_after_welspecs = content[welspecs_start:]
        lines = content_after_welspecs.split('\n')
        
        logger.info(f"Extracted {len(lines)} lines after WELSPECS keyword")
        
        # Skip the WELSPECS line itself
        welspecs_section = []
        for i, line in enumerate(lines):
            if i == 0:  # Skip the WELSPECS keyword line
                continue
                
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Check if we've reached the end of the section
            if line == '/' or (line.startswith('/') and len(line) == 1):
                logger.info(f"Found end of WELSPECS section at line {i}")
                break
                
            # Check if we've reached another keyword
            if line.isupper() and not line.startswith("'"):
                logger.info(f"Found next keyword at line {i}: {line}")
                break
                
            welspecs_section.append(line)
        
        logger.info(f"Extracted {len(welspecs_section)} lines in WELSPECS section")
        
        # Debug: Print first few lines of the section
        for i, line in enumerate(welspecs_section[:min(5, len(welspecs_section))]):
            logger.debug(f"WELSPECS line {i}: {line}")
        
        # Parse each line in the WELSPECS section
        for line_idx, line in enumerate(welspecs_section):
            # Remove trailing slash and comments
            orig_line = line
            if '/' in line:
                line = line.split('/')[0].strip()
            if '--' in line:
                line = line.split('--')[0].strip()
                
            # Skip empty lines
            if not line:
                continue
            
            # FIXED PARSING LOGIC: Parse the line with proper handling of quoted strings
            items = []
            i = 0
            while i < len(line):
                # Skip whitespace
                while i < len(line) and line[i].isspace():
                    i += 1
                
                if i >= len(line):
                    break
                
                # Handle quoted strings
                if line[i] == "'":
                    # Find the closing quote
                    start = i + 1
                    i += 1
                    while i < len(line) and line[i] != "'":
                        i += 1
                    
                    if i < len(line):  # Found closing quote
                        items.append(line[start:i])
                        i += 1  # Skip the closing quote
                    else:
                        # No closing quote found, treat rest as part of string
                        items.append(line[start:])
                else:
                    # Handle non-quoted item
                    start = i
                    while i < len(line) and not line[i].isspace() and line[i] != "'" and line[i] != "/":
                        i += 1
                    
                    # Add the item if not empty
                    if i > start:
                        items.append(line[start:i])
            
            # Debug: show parsed items for each line
            logger.debug(f"Line {line_idx}: Parsed {len(items)} items: {items}")
            
            # Ensure we have enough items
            if len(items) < 6:
                logger.warning(f"Incomplete WELSPECS line: {orig_line}")
                logger.warning(f"Parsed only {len(items)} items: {items}")
                continue
                
            # Extract well name and phase
            well_name = items[0]
            phase = items[5]  # 6th item should be the phase
            
            # Store both original and normalized well names
            normalized_name = normalize_well_name(well_name)
            original_well_names[normalized_name] = well_name
            
            logger.debug(f"Well: {well_name} (normalized: {normalized_name}), Phase: {phase}")
            
            # Categorize well based on phase
            if phase in ['OIL', 'WATER', 'GAS']:
                well_type = 'producer' if phase == 'OIL' else ('water_injector' if phase == 'WATER' else 'gas_injector')
                # Store ONLY the original well name in well_types
                well_types[well_name] = well_type
                # Store normalized name to type mapping separately
                normalized_to_type[normalized_name] = well_type
                logger.debug(f"Set well {well_name} (normalized: {normalized_name}) to type: {well_type}")
            else:
                logger.warning(f"Unknown phase type for well {well_name}: {phase}")
        
        # Add the normalized-to-original mapping to well_types for later use
        well_types['__original_names__'] = original_well_names
        well_types['__normalized_types__'] = normalized_to_type
        
        # Calculate accurate well counts
        unique_wells = set(original_well_names.values())
        original_producers = set(well for well, wtype in well_types.items() 
                            if wtype == 'producer' and well != '__original_names__' and well != '__normalized_types__')
        original_water_injectors = set(well for well, wtype in well_types.items() 
                                  if wtype == 'water_injector' and well != '__original_names__' and well != '__normalized_types__')
        original_gas_injectors = set(well for well, wtype in well_types.items() 
                                if wtype == 'gas_injector' and well != '__original_names__' and well != '__normalized_types__')
        
        # Log accurate counts
        logger.info(f"Parsed {len(unique_wells)} unique wells from WELSPECS")
        
        logger.info(f"Well types: {len(original_producers)} unique producers, "
                   f"{len(original_water_injectors)} unique water injectors, "
                   f"{len(original_gas_injectors)} unique gas injectors")
        
        # Log all producer wells, using original names
        if original_producers:
            logger.info(f"Producer wells from SCHEDULE file (original names): {', '.join(original_producers)}")
        
        return well_types
        
    except Exception as e:
        logger.error(f"Error parsing WELSPECS section: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return well_types

# --- FIX Grid Mapping Helper --- 
def map_active_to_full_grid(active_data: np.ndarray, grid_obj: grid.Grid, nx: int, ny: int, nz: int, default_val: float = 0.0) -> Optional[np.ndarray]:
    """Maps flat data for active cells onto the full 3D grid.

    Args:
        active_data: Flat numpy array containing data for active cells only.
        grid_obj: The resdata Grid object, used for mapping indices.
        nx, ny, nz: Full grid dimensions.
        default_val: Value to fill inactive cells with.

    Returns:
        A 3D numpy array with shape (nz, nx, ny) containing the data mapped
        to the full grid, or None if mapping fails.
    """
    # Check for correct resdata methods
    required_methods = ['get_global_index', 'get_ijk'] # Use correct method names
    if not all(hasattr(grid_obj, method) for method in required_methods):
        missing = [m for m in required_methods if not hasattr(grid_obj, m)]
        logger.error(f"Grid object is missing required mapping methods: {missing}")
        return None
    
    num_active = grid_obj.getNumActive()
    if active_data.size != num_active:
        logger.error(f"Input active_data size ({active_data.size}) does not match grid's active cell count ({num_active}).")
        return None
        
    # Create an empty array for the full grid, ordered (nz, nx, ny)
    full_grid_data = np.full((nz, nx, ny), default_val, dtype=active_data.dtype)

    logger.debug(f"Mapping {num_active} active cells to full grid ({nz}x{nx}x{ny})...")
    mapped_count = 0
    warning_count = 0  # Initialize warning counter
    max_warnings = 10  # Max number of out-of-bounds warnings to show
    try:
        for active_idx in range(num_active):
            global_idx = grid_obj.get_global_index(active_index=active_idx)
            # Get native indices (i corresponds to nx, j to ny, k to nz)
            i_nat, j_nat, k_nat = grid_obj.get_ijk(global_index=global_idx)
            
            # Check bounds using native indices against native dimensions
            if 0 <= i_nat < nx and 0 <= j_nat < ny and 0 <= k_nat < nz:
                 # Write to the target array using (k, i, j) order for (nz, nx, ny) shape
                 full_grid_data[k_nat, i_nat, j_nat] = active_data[active_idx]
                 mapped_count += 1
            else:
                 # Log warning but limit the number of messages
                 if warning_count < max_warnings:
                     logger.warning(f"Native index ({i_nat},{j_nat},{k_nat}) from global_idx {global_idx} is out of grid bounds ({nx},{ny},{nz}).")
                 elif warning_count == max_warnings:
                      logger.warning(f"Further out-of-bounds warnings suppressed...")
                 warning_count += 1
        logger.debug(f"Successfully mapped {mapped_count}/{num_active} cells.")
        if mapped_count != num_active:
             logger.warning(f"Mismatch in mapped cell count ({mapped_count}) vs expected ({num_active}). Total out-of-bounds warnings: {warning_count}")
        return full_grid_data
    except Exception as e:
        logger.error(f"Error during active cell mapping: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
# --- END Grid Mapping Helper ---

class EclipseReader:
    """Reader for Eclipse simulation files. Can handle GRID, INIT, and UNRST (via resdata)."""
    
    def __init__(self, filepath: Union[str, Path]) -> None:
        """Initialize the reader.
        
        Args:
            filepath: Path to the Eclipse file (GRID, INIT, or UNRST)
        """
        self.filepath = Path(filepath)
        # self.actnum = None  # REMOVED: Will store ACTNUM data if loaded
        self.grid_obj = None # Store grid object if read
        self.rst_file = None # Store ResdataFile object for UNRST
        self.report_dates = None
        self.available_steps = None
        self.nx = None
        self.ny = None
        self.nz = None
        logger.info(f"Initializing EclipseReader with file: {self.filepath}")
        
    def read_grid(self, apply_mapaxes=True) -> Optional[grid.Grid]:
        """Read Eclipse grid file.
        
        Returns:
            Grid object containing the reservoir grid, or None on failure.
        """
        logger.info(f"Reading grid file: {self.filepath}")
        try:
            grid_obj = grid.Grid(str(self.filepath))
            logger.info("Successfully read grid file")
            self.nx, self.ny, self.nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
            logger.info(f"Grid dimensions: {self.nx}x{self.ny}x{self.nz}")
            self.grid_obj = grid_obj # Store for potential later use
            return grid_obj
        except Exception as e:
            logger.error(f"Error reading grid file {self.filepath}: {str(e)}")
            # Return None instead of raising, allows calling code to handle missing files more gracefully
            return None
            
    def read_well_types_from_schedule(self, schedule_file: Union[str, Path]) -> Dict[str, str]:
        """Read well types from SCHEDULE file by parsing WELSPECS sections.
        
        Args:
            schedule_file: Path to the SCHEDULE file
            
        Returns:
            Dictionary mapping well names to their types (producer, water_injector, gas_injector)
        """
        try:
            schedule_path = Path(schedule_file)
            if not schedule_path.exists():
                logger.warning(f"SCHEDULE file not found at {schedule_path}")
                return {}
                
            logger.info(f"Reading well types from SCHEDULE file: {schedule_path}")
            
            # Read the SCHEDULE file content
            with open(schedule_path, 'r') as f:
                content = f.read()
                
            logger.info(f"SCHEDULE file size: {len(content)} bytes")
            
            # Parse the WELSPECS section to get well types
            well_types = parse_welspecs_section(content)
            
            logger.info(f"Found {len(well_types)} wells in SCHEDULE file")
            
            return well_types
            
        except Exception as e:
            logger.error(f"Error reading well types from SCHEDULE file {schedule_file}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
            
    def find_schedule_file(self, base_dir: Union[str, Path] = None) -> Optional[Path]:
        """Find SCHEDULE file in standard locations relative to a base directory.
        
        Args:
            base_dir: Base directory to search from (default: directory of current file)
            
        Returns:
            Path to SCHEDULE file if found, None otherwise
        """
        if base_dir is None:
            base_dir = self.filepath.parent
        else:
            base_dir = Path(base_dir)
            
        logger.info(f"Looking for SCHEDULE file in directory: {base_dir}")
        
        # Common schedule file names and locations
        possible_paths = [
            base_dir / "include" / "js_final_hist.sch",  # JOHAN_SVERDRUP specific
            base_dir / "include" / "schedule.sch",
            base_dir / "include" / "SCHEDULE.sch",
            base_dir / "SCHEDULE.sch",
            base_dir / "schedule.sch",
        ]
        
        # First try the standard names
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found SCHEDULE file at: {path}")
                return path
                
        # If not found, try searching for any .sch file in the include directory
        include_dir = base_dir / "include"
        if include_dir.exists():
            sch_files = list(include_dir.glob("*.sch"))
            if sch_files:
                logger.info(f"Found potential SCHEDULE file: {sch_files[0]}")
                return sch_files[0]
                
        # Finally, try searching for any .sch file in the base directory
        sch_files = list(base_dir.glob("*.sch"))
        if sch_files:
            logger.info(f"Found potential SCHEDULE file: {sch_files[0]}")
            return sch_files[0]
            
        logger.warning(f"No SCHEDULE file found in standard locations")
        return None

    def read_init(self, grid_obj: Optional[grid.Grid] = None) -> Optional[resfile.ResdataInitFile]:
        """Read Eclipse INIT file.
        
        Args:
            grid_obj: Grid object required for INIT file initialization.
                      If None, attempts to use a previously read grid object.
            
        Returns:
            ResdataInitFile object containing the initialization data, or None on failure.
        """
        if grid_obj is None:
            grid_obj = self.grid_obj # Use stored grid if available

        if grid_obj is None:
            logger.error(f"Grid object required but not provided or previously read for INIT file {self.filepath}")
            return None

        # --- Store grid object and dimensions locally --- 
        self.grid_obj = grid_obj # Store the grid object itself
        self.nx = grid_obj.getNX()
        self.ny = grid_obj.getNY()
        self.nz = grid_obj.getNZ()
        if not all([self.nx, self.ny, self.nz]):
             logger.error(f"Could not determine grid dimensions from provided grid object for {self.filepath}")
             return None
        logger.info(f"Using grid dimensions for INIT processing: {self.nx}x{self.ny}x{self.nz}")
        # --- End Store --- 

        logger.info(f"Reading INIT file: {self.filepath}")
        try:
            init = resfile.ResdataInitFile(grid_obj, str(self.filepath))
            logger.info("Successfully read INIT file")
            
            return init
        except Exception as e:
            logger.error(f"Error reading INIT file {self.filepath}: {str(e)}")
            return None # Return None on failure
            
    def open_restart_file(self, grid_obj_passed: Optional[grid.Grid] = None) -> bool:
        """Opens the UNRST file using ResdataFile and stores relevant info.
        
        Args:
            grid_obj: Grid object required for restart file initialization.
                      If None, attempts to use a previously read grid object.
            
        Returns:
            True if the file was opened successfully, False otherwise.
        """
        if self.rst_file is not None:
            logger.warning(f"Restart file {self.filepath} seems to be already open. Closing first.")
            self.close_restart_file()

        if grid_obj_passed is None:
            grid_obj_passed = self.grid_obj # Use stored grid if available

        if grid_obj_passed is None:
            logger.error(f"Grid object required but not provided or previously read for restart file {self.filepath}")
            return False

        # Ensure grid dimensions are known
        if not all([self.nx, self.ny, self.nz]):
            self.nx, self.ny, self.nz = grid_obj_passed.getNX(), grid_obj_passed.getNY(), grid_obj_passed.getNZ()
            if not all([self.nx, self.ny, self.nz]):
                 logger.error(f"Could not determine grid dimensions for {self.filepath}")
                 return False

        logger.debug(f"Opening restart file: {self.filepath} using ResdataRestartFile")
        original_path_str = str(self.filepath)
        logger.debug(f"Calling: resfile.ResdataRestartFile(grid_obj(RELOADED)='{type(grid_obj_passed)}', path='{original_path_str}')")
        
        # --- DETAILED PRE-CALL LOGGING ---
        logger.debug(f"    Attempting ResdataRestartFile call:")
        logger.debug(f"    CWD: {Path.cwd()}")
        logger.debug(f"    Resolved Path: {original_path_str} (Type: {type(original_path_str)})")
        logger.debug(f"    Grid Object: {grid_obj_passed} (Type: {type(grid_obj_passed)})")
        
        # Explicit check of grid_obj validity right before use
        if grid_obj_passed is None:
            logger.error("    Grid object is None immediately before ResdataRestartFile call!")
            return False
        if not hasattr(grid_obj_passed, 'nx') or not hasattr(grid_obj_passed, 'getNX'):
             logger.error("    Grid object appears invalid (missing key attributes) before ResdataRestartFile call!")
             return False
        logger.debug(f"    Grid Object Dims Check: {grid_obj_passed.getNX()}x{grid_obj_passed.getNY()}x{grid_obj_passed.getNZ()}")
        # --- END PRE-CALL LOGGING ---
        
        # --- Store the validated grid object internally ---
        self.grid_obj = grid_obj_passed
        # --- End Store --- 
        
        try:
            # Ensure the file exists right before opening
            if not self.filepath.exists():
                logger.error(f"File does not exist at path: {self.filepath}")
                return False           
            # Use ResdataRestartFile with the INTERNALLY RELOADED grid object
            self.rst_file = resfile.ResdataRestartFile(grid_obj_passed, original_path_str)
            self.available_steps = self.rst_file.report_steps
            logger.info(f"Successfully opened restart file. Available steps: {self.available_steps}")
            return True
        except Exception as e:
            logger.error(f"Error opening restart file {self.filepath}: {str(e)}")
            self.rst_file = None
            return False

    def read_initial_conditions(self) -> Optional[Dict[str, np.ndarray]]:
        """Reads initial conditions (step 0 or first available) from the opened UNRST file.
        
        Returns:
            Dictionary containing initial condition data ('Pini', 'Sini', etc.), or None on failure.
        """
        if self.rst_file is None:
            logger.error("Restart file must be opened using open_restart_file() before reading initial conditions.")
            return None

        initial_data = {}
        initial_step = 0
        if not self.rst_file.has_report_step(initial_step):
            if self.available_steps:
                initial_step = self.available_steps[0]
                logger.warning(f"Report step 0 not found. Using first available step ({initial_step}) for initial conditions.")
            else:
                logger.error("No report steps found in UNRST file to get initial conditions.")
                return None

        # Find the index of the initial_step within the available steps
        # This is needed to select the correct slice if iget_kw returns all timesteps
        initial_step_idx_in_available = -1
        if self.available_steps is not None:
            try:
                initial_step_idx_in_available = self.available_steps.index(initial_step)
            except ValueError:
                logger.error(f"Initial step {initial_step} not found in available steps {self.available_steps}. Cannot read initial conditions.")
                return None
        else:
            logger.error("Available steps list not populated. Cannot read initial conditions.")
            return None

        # --- Calculate 1-based access index --- 
        access_index = initial_step_idx_in_available + 1
        logger.info(f"Loading initial conditions from report step {initial_step} (using 1-based access index {access_index})")
        try:
            # Define keywords to read for initial conditions
            initial_keywords = {
                'PRESSURE': 'PRESSURE',
                'SWAT': 'SWAT',
                # Add 'SGAS': 'SGAS', 'SOIL': 'SOIL' here if needed
            }
            
            for data_key, keyword in initial_keywords.items():
                # --- Use report number (initial_step) for has_kw --- 
                if self.rst_file.has_kw(keyword, initial_step):
                    active_cell_data_slice = None
                    full_grid_mapped = None
                    raw_kw_data = None # Variable to hold direct output 
                    np_raw_data = None # Variable to hold numpy converted data
                    
                    # --- Use 1-based access_index for dictionary access --- 
                    try:
                        logger.debug(f"Attempting direct access for {keyword}, step {initial_step} using index {access_index}: rst_file['{keyword}'][{access_index}]")
                        raw_kw_data = self.rst_file[keyword][access_index]
                        logger.debug(f"Direct access successful for {keyword}, step {initial_step}. Type: {type(raw_kw_data)}")
                    except (KeyError, IndexError):
                         logger.warning(f"Key '{keyword}' or index {access_index} (for step {initial_step}) not found using direct access.")
                         raw_kw_data = None
                    except AttributeError as ae:
                         logger.error(f"AttributeError (likely _copyc) during direct access for {keyword} index {access_index}: {ae}", exc_info=False)
                         raw_kw_data = None
                    except Exception as e:
                        logger.error(f"Unexpected error during direct access for {keyword} index {access_index}: {e}", exc_info=True)
                        raw_kw_data = None
                    # --- End Direct Access --- 

                    # Proceed only if direct access worked
                    if raw_kw_data is not None:
                        # --- Extract numpy data (no change needed here) --- 
                        try:
                            if hasattr(raw_kw_data, 'numpyView'):
                                np_raw_data = raw_kw_data.numpyView()
                            elif isinstance(raw_kw_data, (list, tuple)):
                                np_raw_data = np.array(raw_kw_data)
                            elif isinstance(raw_kw_data, np.ndarray):
                                np_raw_data = raw_kw_data
                            else:
                                logger.warning(f"Unexpected data type {type(raw_kw_data)} from direct access for {keyword} index {access_index}")
                        except Exception as np_err:
                             logger.error(f"Error converting data to numpy for {keyword} index {access_index}: {np_err}", exc_info=True)
                             np_raw_data = None
                             
                        # --- Data should be 1D now, no time slicing needed --- 
                        if np_raw_data is not None:
                            if np_raw_data.ndim == 1:
                                active_cell_data_slice = np_raw_data
                            else:
                                logger.warning(f"Expected 1D array after access by index for {keyword} index {access_index}, but got shape {np_raw_data.shape}. Cannot map.")
                        else:
                             logger.warning(f"Could not convert/extract numpy array for {keyword} index {access_index}")

                    # --- Mapping (no change needed here) --- 
                    if active_cell_data_slice is not None:
                        # Assuming grid_obj was stored during reader initialization or grid read
                        if self.grid_obj is None:
                             logger.error("Cannot map active cells: Grid object not available in EclipseReader.")
                             continue # Skip keyword
                        
                        full_grid_mapped = map_active_to_full_grid(
                            active_cell_data_slice,
                            self.grid_obj,
                            self.nx, self.ny, self.nz
                        )
                        
                        if full_grid_mapped is not None:
                            initial_data[data_key] = full_grid_mapped
                            logger.info(f"Successfully read and mapped {keyword} for initial conditions.")
                        else:
                            logger.error(f"Failed to map active cells for {keyword} at initial step {initial_step}")
                    # --- End Mapping --- 
                else:
                    # --- Use report number (initial_step) for logging --- 
                    logger.warning(f"Keyword '{keyword}' not found via has_kw for initial step {initial_step}")

            logger.info(f"Successfully attempted to read initial conditions from step {initial_step}")
            return initial_data
        except Exception as e:
            logger.error(f"Error reading initial conditions from step {initial_step}: {str(e)}")
            return None

    def read_dynamic_steps(self, report_steps: List[int]) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """Reads specified dynamic report steps from the opened UNRST file.
        
        Args:
            report_steps: List of integer report steps to read.
            
        Returns:
            A tuple containing:
            - Dictionary of dynamic data (e.g., 'Pressure', 'Water_saturation') with shape (n_steps, nx, ny, nz).
            - Numpy array of corresponding simulation times in days relative to step 0.
            Returns None if the restart file is not open or on critical error.
        """
        if self.rst_file is None or self.available_steps is None:
            logger.error("Restart file must be opened using open_restart_file() before reading dynamic steps.")
            return None

        logger.info(f"Loading data for {len(report_steps)} selected report steps: {report_steps}")
        dynamic_data = {
            'PRESSURE': [], 'SWAT': [], 'SGAS': [], 'SOIL': []
        }
        step_dates_collected = [] # Store successfully read dates (if available)
        valid_steps_read_indices = [] # Store indices of steps successfully read
        successfully_read_steps = [] # Store the actual step numbers read

        # --- Access report_dates with error handling ---
        report_dates = None
        initial_date = None
        try:
            report_dates = self.rst_file.report_dates
            logger.info(f"Successfully accessed report dates (count: {len(report_dates) if report_dates else 'N/A'})")
            # Find initial date for time calculation (only if dates were accessed)
            initial_step_for_date = 0
            if 0 not in self.available_steps and self.available_steps:
                 initial_step_for_date = self.available_steps[0]
            initial_date_index = self.available_steps.index(initial_step_for_date)
            initial_date = report_dates[initial_date_index]
            logger.info(f"Determined initial date: {initial_date} from step {initial_step_for_date}")
        except OSError as e:
            logger.warning(f"OSError accessing report dates: {e}. Will use report step numbers for Time array.")
            report_dates = None # Ensure it's None if error occurred
            initial_date = None
        except Exception as e:
            logger.warning(f"Unexpected error accessing report dates: {e}. Will use report step numbers for Time array.")
            report_dates = None
            initial_date = None
        # --- End date access handling ---
        
        # --- Loop through steps --- 
        for idx, step in enumerate(report_steps):
            # Find the index of this step within the available steps
            step_idx_in_available = -1
            if self.available_steps is not None:
                try:
                    step_idx_in_available = self.available_steps.index(step)
                except ValueError:
                    logger.warning(f"Requested step {step} not in available steps {self.available_steps}. Skipping.")
                    # Append None to keep lists aligned for filtering later
                    dynamic_data['PRESSURE'].append(None)
                    dynamic_data['SWAT'].append(None)
                    dynamic_data['SGAS'].append(None)
                    dynamic_data['SOIL'].append(None)
                    continue # Skip to next requested step
            else:
                 logger.error("Available steps list not populated. Cannot process dynamic steps.")
                 return None # Critical error
                 
            logger.debug(f"Reading data for report step {step} (index {step_idx_in_available} in available steps)")
            step_read_success = False
            # --- Calculate 1-based access index based on position in available_steps --- 
            access_index = step_idx_in_available + 1 
            try:
                # --- Refined Keyword Processing within Loop --- 
                current_step_data = {}
                keywords_to_process = ['PRESSURE', 'SWAT', 'SGAS', 'SOIL']
                
                for key in keywords_to_process:
                    active_cell_data_slice = None
                    full_grid_mapped = None
                    raw_kw_data = None # Variable to hold direct output of dictionary access
                    np_raw_data = None # Variable to hold numpy converted data
                    
                    # --- Use direct dictionary access with CORRECT 1-based positional index --- 
                    try:
                        # --- Use report number (step) for checking key existence --- 
                        if key in self.rst_file:
                             logger.debug(f"Attempting direct access for {key}, step {step} using 1-based positional index {access_index}: rst_file['{key}'][{access_index}]")
                             raw_kw_data = self.rst_file[key][access_index] # <-- Use CORRECT access_index
                             logger.debug(f"Direct access successful for {key}, step {step}. Type: {type(raw_kw_data)}")
                        else:
                             logger.debug(f"Keyword {key} not found in rst_file keys for step {step}")
                             raw_kw_data = None # Treat as missing
                    except (KeyError, IndexError):
                         # Log using both step and access_index for clarity
                         logger.warning(f"Key '{key}' or 1-based index {access_index} (for step {step}) not found using direct access.")
                         raw_kw_data = None
                    except AttributeError as ae:
                         logger.error(f"AttributeError (likely _copyc) during direct access for {key} index {access_index}: {ae}", exc_info=False)
                         raw_kw_data = None
                    except Exception as e:
                        logger.error(f"Unexpected error during direct access for {key} index {access_index}: {e}", exc_info=True)
                        raw_kw_data = None
                    # --- End Direct Access --- 

                    # Proceed only if direct access didn't error and returned something
                    if raw_kw_data is not None:
                        # --- Extract numpy data (no change) --- 
                        try:
                            if hasattr(raw_kw_data, 'numpyView'):
                                np_raw_data = raw_kw_data.numpyView()
                            elif isinstance(raw_kw_data, (list, tuple)):
                                np_raw_data = np.array(raw_kw_data)
                            elif isinstance(raw_kw_data, np.ndarray):
                                np_raw_data = raw_kw_data
                            else:
                                logger.warning(f"Unexpected data type {type(raw_kw_data)} from direct access for {key} index {access_index}")
                        except Exception as np_err:
                             logger.error(f"Error converting data to numpy for {key} index {access_index}: {np_err}", exc_info=True)
                             np_raw_data = None
                             
                        # --- Data should be 1D now, no time slicing needed --- 
                        if np_raw_data is not None:
                            if np_raw_data.ndim == 1:
                                active_cell_data_slice = np_raw_data
                            else:
                                logger.warning(f"Expected 1D array after access by index for {key} index {access_index}, but got shape {np_raw_data.shape}. Cannot map.")
                        else:
                             logger.warning(f"Could not convert/extract numpy array for {key} index {access_index}")

                    # --- Mapping (no change) --- 
                    if active_cell_data_slice is not None:
                        if self.grid_obj:
                            full_grid_mapped = map_active_to_full_grid(
                                active_cell_data_slice, self.grid_obj, self.nx, self.ny, self.nz
                            )
                            if full_grid_mapped is None:
                                 logger.error(f"Mapping failed for {key} step {step}")
                        else:
                             logger.error(f"Cannot map {key} step {step}: Grid object missing.")
                              
                    # Store the final mapped data (or None if any step failed)
                    current_step_data[key] = full_grid_mapped 
                    # --- End Refined Keyword Processing --- 
                    
                # Append results for this step (will contain None if any sub-step failed)
                dynamic_data['PRESSURE'].append(current_step_data['PRESSURE'])
                dynamic_data['SWAT'].append(current_step_data['SWAT'])
                dynamic_data['SGAS'].append(current_step_data['SGAS'])
                dynamic_data['SOIL'].append(current_step_data['SOIL'])
                
                # Store date if available (assuming dates were successfully read earlier)
                if report_dates is not None and initial_date is not None:
                    try:
                        date_index = self.available_steps.index(step)
                        step_dates_collected.append(report_dates[date_index])
                    except (ValueError, IndexError):
                         logger.warning(f"Could not find date for step {step}.")
                         step_dates_collected.append(None)
                
                valid_steps_read_indices.append(idx)
                successfully_read_steps.append(step)
                step_read_success = True # Mark step as attempted
                
            except Exception as loop_err: # Catch unexpected errors within the step loop
                logger.error(f"Critical error processing dynamic step {step}: {loop_err}", exc_info=True)
                # Append Nones to maintain list structure if error occurred mid-step
                dynamic_data['PRESSURE'].append(None)
                dynamic_data['SWAT'].append(None)
                dynamic_data['SGAS'].append(None)
                dynamic_data['SOIL'].append(None)
                if report_dates is not None: step_dates_collected.append(None)
        # --- End loop ---

        # --- Filter and Stack MAPPED data --- 
        final_dynamic_data = {}
        num_valid_steps = len(successfully_read_steps)
        if num_valid_steps == 0:
             logger.warning("No dynamic steps were successfully read and processed.")
             return {}, np.array([], dtype=np.float32) # Return empty results
        
        for key in ['PRESSURE', 'SWAT', 'SGAS', 'SOIL']:
            # Get the data collected for this key (list of 3D arrays or Nones)
            value_list_for_key = dynamic_data[key]
            # Filter out the None values corresponding to failed reads/maps for this step/key
            valid_values = [value_list_for_key[i] for i in range(len(report_steps)) if i in valid_steps_read_indices and value_list_for_key[i] is not None]
            
            if valid_values:
                # Check if number of valid steps matches expectation
                if len(valid_values) != num_valid_steps:
                    logger.warning(f"Mismatch count for {key}: Expected {num_valid_steps} valid steps, found {len(valid_values)} arrays.")
                    # Decide how to handle: skip key, pad, error? Let's try to stack anyway.
                try:
                    # Stack along a new time dimension (axis=0)
                    stacked_data = np.stack(valid_values, axis=0)
                    # Transpose to (time_dim, nz, nx, ny) - assuming map_active returns (nz, nx, ny)
                    final_dynamic_data[key] = stacked_data # Shape is now (time, nz, nx, ny)
                    logger.info(f"Stacked {key} data with final shape {final_dynamic_data[key].shape}")
                except ValueError as e:
                    logger.error(f"Error stacking {key}: {e}. Shapes: {[v.shape for v in valid_values]}")
            else:
                logger.warning(f"No valid data found for dynamic property: {key} after processing.")
        # --- End Filter and Stack --- 
        
        # --- Calculate Time array (Date-based or Step-based) --- 
        time_array = None
        valid_step_dates = [d for d in step_dates_collected if d is not None]

        if report_dates is not None and initial_date is not None and len(valid_step_dates) == len(successfully_read_steps):
            # Use dates if successfully read for all valid steps
            try:
                time_deltas = [(date - initial_date).total_seconds() / (24 * 3600) for date in valid_step_dates]
                time_array = np.array(time_deltas, dtype=np.float32)
                logger.info(f"Calculated Time array based on report dates (days from {initial_date}). Length: {len(time_array)}")
            except Exception as calc_e:
                logger.warning(f"Error calculating time deltas from dates: {calc_e}. Falling back to step numbers.")
                time_array = None # Fallback
        
        if time_array is None:
            # Fallback: Use the step numbers that were successfully read
            time_array = np.array(successfully_read_steps, dtype=np.float32)
            logger.warning(f"Using report step numbers for Time array. Length: {len(time_array)}")
        # --- End Time array calculation ---
        
        return final_dynamic_data, time_array

    def close_restart_file(self) -> None:
        """Closes the ResdataFile object if it's open."""
        if self.rst_file is not None:
            try:
                self.rst_file.close()
                logger.info(f"Closed restart file: {self.filepath}")
            except Exception as e:
                logger.error(f"Error closing restart file {self.filepath}: {str(e)}")
            finally:
                self.rst_file = None
                self.report_dates = None
                self.available_steps = None

    def read_summary(self, filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Read summary data from SMSPEC and UNSMRY files.
        
        Args:
            filepath: Path to SMSPEC or UNSMRY file (optional). 
                     If None, uses the path provided during initialization.
                     
        Returns:
            DataFrame containing summary data
        """
        path = Path(filepath) if filepath else Path(self.filepath)
        logger.info(f"Reading summary data from: {path}")
        
        try:
            # For SMSPEC or UNSMRY, we need the case prefix (without extension)
            # This works because resdata automatically looks for companion files
            if path.suffix.lower() in ['.smspec', '.sms', '.unsmry', '.usy']:
                # Extract the case prefix (filename without extension)
                case_prefix = str(path.with_suffix(''))
                logger.info(f"Using case prefix: {case_prefix}")
            else:
                # Assume it's a case prefix already
                case_prefix = str(path)
                
            # Create a Summary object
            summary_obj = summary.Summary(case_prefix)
            
            # Check if summary was loaded successfully
            if not summary_obj:
                raise ValueError(f"Failed to load summary from {case_prefix}")
                
            # Get all vectors available in the summary
            vector_list = summary_obj.keys()
            logger.info(f"Found {len(vector_list)} vectors in summary")
            
            # Extract vectors into a dataframe
            data = {}
            for vector in vector_list:
                try:
                    data[vector] = summary_obj.numpy_vector(vector)
                except Exception as e:
                    logger.warning(f"Failed to extract vector {vector}: {str(e)}")
            
            # Create a DataFrame with all vectors
            df = pd.DataFrame(data)
            
            # Add time vector if available
            if 'TIME' in vector_list:
                df['TIME'] = summary_obj.numpy_vector('TIME')
            elif 'DAYS' in vector_list:
                df['DAYS'] = summary_obj.numpy_vector('DAYS')
                
            logger.info(f"Successfully read summary data with {len(df.columns)} vectors and {len(df)} time steps")
            return df
            
        except Exception as e:
            logger.error(f"Error reading summary data: {str(e)}")
            raise

    def read_summary_vectors(self, filepath: Optional[Union[str, Path]] = None) -> List[str]:
        """Read only vector names from SMSPEC file without loading data.
        
        Args:
            filepath: Path to SMSPEC file (optional). 
                     If None, uses the path provided during initialization.
                     
        Returns:
            List of vector names available in the summary
        """
        path = Path(filepath) if filepath else Path(self.filepath)
        logger.info(f"Reading vector names from: {path}")
        
        try:
            # For SMSPEC, we need the case prefix (without extension)
            if path.suffix.lower() in ['.smspec', '.sms']:
                # Extract the case prefix (filename without extension)
                case_prefix = str(path.with_suffix(''))
            else:
                # Assume it's a case prefix already
                case_prefix = str(path)
                
            # Create a Summary object
            summary_obj = summary.Summary(case_prefix)
            
            # Check if summary was loaded successfully
            if not summary_obj:
                raise ValueError(f"Failed to load summary from {case_prefix}")
                
            # Get all vectors available in the summary without loading data
            vector_list = summary_obj.keys()
            logger.info(f"Found {len(vector_list)} vectors in summary")
            
            # Return the list of vector names
            return vector_list
            
        except Exception as e:
            logger.error(f"Error reading summary vectors: {str(e)}")
            raise
    
    def get_property(self, file_obj, property_name):
        """
        Extracts a property from INIT or restart file objects and handles reshaping.

        Args:
            file_obj: ResdataInitFile or ResdataFile object
            property_name: Name of the property to extract (e.g. 'PORO', 'PERMX', 'PRESSURE')

        Returns:
            np.ndarray: Property data reshaped to (nz, nx, ny) or None if extraction failed
        """
        try:
            # No grid dimensions available
            if not all([self.nx, self.ny, self.nz]):
                logger.error(f"Grid dimensions not set. Cannot extract property: {property_name}")
                return None

            if property_name not in file_obj.keys():
                logger.warning(f"Property '{property_name}' not found in keys: {list(file_obj.keys())}")
                return None
                
            # Extract using direct array access with [0] for ResdataInitFile
            if isinstance(file_obj, resfile.ResdataInitFile):
                logger.debug(f"Extracting property: {property_name} from INIT object using index [0]")
                try:
                    prop_obj = file_obj[property_name][0]  # Access first array for INIT properties
                    logger.debug(f"  Direct access [0] successful. Type: {type(prop_obj)}")
                except Exception as e:
                    logger.error(f"Error extracting property '{property_name}' from INIT: {e}")
                    return None
            # For restart files (ResdataFile)
            elif isinstance(file_obj, resfile.ResdataFile):
                report_step = self.report_steps[self.current_step] if hasattr(self, 'current_step') else -1
                logger.debug(f"Extracting property: {property_name} from restart for report step {report_step}")
                try:
                    prop_obj = file_obj[property_name, report_step]  # Access based on report step for restart
                    logger.debug(f"  Access for step {report_step} successful. Type: {type(prop_obj)}")
                except Exception as e:
                    logger.error(f"Error extracting property '{property_name}' from restart: {e}")
                    return None
            else:
                logger.error(f"Unsupported file object type: {type(file_obj)}")
                return None

            # --- Convert to numpy and reshape for all property types ---
            try:
                # Convert to numpy array
                prop_data = np.array(prop_obj)
                logger.debug(f"  Raw numpy data shape: {prop_data.shape}, dtype: {prop_data.dtype}")
                
                # For 1D arrays (regardless of exact size), reshape using grid dimensions
                if prop_data.ndim == 1:
                    # Handle active cell data (with mapping) if we have grid object
                    if self.grid_obj and hasattr(self.grid_obj, 'getNumActive'):
                        try:
                            logger.debug(f"  Attempting to map active cells to grid...")
                            active_cells = self.grid_obj.getNumActive()
                            logger.debug(f"  Active cells in grid: {active_cells}")
                            
                            # For active cell arrays that match grid's active count
                            if prop_data.size == active_cells:
                                logger.debug(f"  Data size matches active cells, mapping to full grid")
                                return  map_active_to_full_grid(
                                    prop_data, self.grid_obj, self.nx, self.ny, self.nz
                                )
                        except Exception as e:
                            logger.warning(f"  Active cell mapping failed: {e}")
                    
                    # If not mapped as active cells, try direct reshaping to full grid
                    try:
                        # Check if size matches full grid
                        full_size = self.nx * self.ny * self.nz
                        if prop_data.size == full_size:
                            logger.debug(f"  Reshaping data to ({self.nx}, {self.ny}, {self.nz}) then transposing")
                            reshaped = prop_data.reshape((self.nx, self.ny, self.nz), order='F')
                            # Transpose to target shape (nz, nx, ny)
                            result = np.transpose(reshaped, (2, 0, 1))
                            logger.debug(f"  Final shape after reshape: {result.shape}")
                            return result
                        else:
                            # Special fallback for INIT properties with unexpected size
                            if isinstance(file_obj, resfile.ResdataInitFile):
                                logger.warning(f"  Property '{property_name}' from INIT has size {prop_data.size}, "
                                             f"which doesn't match full grid size {full_size}.")
                                logger.warning(f"  Attempting best effort reshape...")
                                # Try to reshape assuming it's a different subset of cells
                                try:
                                    # Try trimming or padding if needed
                                    adjusted_data = prop_data
                                    if prop_data.size > full_size:
                                        logger.warning(f"  Trimming data from {prop_data.size} to {full_size} elements")
                                        adjusted_data = prop_data[:full_size]
                                    elif prop_data.size < full_size:
                                        logger.warning(f"  Padding data from {prop_data.size} to {full_size} elements with zeros")
                                        adjusted_data = np.zeros(full_size, dtype=prop_data.dtype)
                                        adjusted_data[:prop_data.size] = prop_data
                                    
                                    reshaped = adjusted_data.reshape((self.nx, self.ny, self.nz), order='F')
                                    result = np.transpose(reshaped, (2, 0, 1))
                                    logger.info(f"  Successfully reshaped '{property_name}' with best effort approach")
                                    return result
                                except Exception as reshape_err:
                                    logger.error(f"  Best effort reshape failed: {reshape_err}")
                    except Exception as e:
                        logger.error(f"  Direct reshape failed: {e}")
                # For already 3D arrays
                elif prop_data.ndim == 3:
                    # Check if shapes match expected dimensions
                    if prop_data.shape == (self.nx, self.ny, self.nz):
                        logger.debug(f"  Array already 3D with shape {prop_data.shape}, transposing to (nz,nx,ny)")
                        return np.transpose(prop_data, (2, 0, 1))
                    elif prop_data.shape == (self.nz, self.nx, self.ny):
                        logger.debug(f"  Array already in target shape {prop_data.shape}")
                        return prop_data
                    else:
                        logger.warning(f"  3D array has unexpected shape {prop_data.shape}, expected either "
                                     f"({self.nx}, {self.ny}, {self.nz}) or ({self.nz}, {self.nx}, {self.ny})")
                
                # If we get here, we couldn't handle the shape properly
                logger.warning(f"  Property '{property_name}' has shape {prop_data.shape} that couldn't be processed")
                return None
                
            except Exception as e:
                logger.error(f"Error processing property '{property_name}': {e}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error extracting property '{property_name}': {e}")
            return None
            
    def count_active_cells(self, grid_obj) -> int:
        """Count active cells in the grid.
        
        Args:
            grid_obj: Grid object
            
        Returns:
            Number of active cells
        """
        try:
            if hasattr(grid_obj, 'getNumActive'):
                # Direct method available
                return grid_obj.getNumActive()
            else:
                # Count manually
                nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
                active_count = 0
                
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            if self.is_cell_active(grid_obj, i, j, k):
                                active_count += 1
                                
                return active_count
        except Exception as e:
            logger.error(f"Error counting active cells: {str(e)}")
            return 0
            
    def get_property_info(self, file_obj, property_name: str) -> dict:
        """Get detailed information about a property.
        
        Args:
            file_obj: ResdataInitFile or ResdataRestartFile object
            property_name: Name of the property to retrieve
            
        Returns:
            Dictionary with property metadata
        """
        try:
            # Check if property exists
            if property_name not in file_obj:
                return {"exists": False, "error": f"Property {property_name} not found"}
            
            # Get basic information
            info = {"exists": True, "name": property_name}
            
            # Try to get property data
            property_data = file_obj[property_name]
            if not property_data:
                info["error"] = f"Failed to access property data"
                return info
            
            # Try to get more information
            if hasattr(property_data, "getSize"):
                info["size"] = property_data.getSize()
            
            if hasattr(property_data, "getType"):
                info["type"] = property_data.getType()
            
            # Try to get values
            try:
                data = np.array(property_data)
                info["shape"] = data.shape
                info["ndim"] = data.ndim
                info["dtype"] = str(data.dtype)
                
                if data.size > 0:
                    info["min"] = float(np.nanmin(data))
                    info["max"] = float(np.nanmax(data))
                    info["mean"] = float(np.nanmean(data))
                
                # Check if it's likely a cell property
                if hasattr(file_obj, 'grid'):
                    grid = file_obj.grid
                    active_size = grid.getNumActive()
                    global_size = grid.getGlobalSize()
                    
                    if data.size == active_size:
                        info["property_type"] = "active_cells"
                    elif data.size == global_size:
                        info["property_type"] = "global_cells"
            except Exception as e:
                info["conversion_error"] = str(e)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting property info for {property_name}: {str(e)}")
            return {"exists": False, "error": str(e)}
    
    def get_cell_corners(self, grid_obj, i: int, j: int, k: int) -> np.ndarray:
        """Get the corner points of a specific cell.
        
        Args:
            grid_obj: Grid object
            i: I-index of the cell
            j: J-index of the cell
            k: K-index of the cell
            
        Returns:
            Array of 8 corner points, each with (x,y,z) coordinates
        """
        try:
            corners = np.zeros((8, 3))
            for corner_idx in range(8):
                # Get corner using the right method depending on grid API
                try:
                    # First try the cell_corners method
                    if hasattr(grid_obj, 'get_cell_corner'):
                        # New API
                        x, y, z = grid_obj.get_cell_corner(corner_idx, i, j, k)
                    elif hasattr(grid_obj, 'getCellCorner'):
                        # Old API - pass ijk as a tuple without the parameter name
                        x, y, z = grid_obj.getCellCorner(corner_idx, (i, j, k))
                    elif hasattr(grid_obj, 'cell_corners'):
                        # Alternative API
                        corners = grid_obj.cell_corners(i, j, k)
                        break
                    else:
                        # Get global index and try that approach
                        global_idx = grid_obj.get_global_index(i, j, k)
                        x, y, z = grid_obj.get_cell_corner(corner_idx, global_index=global_idx)
                    
                    corners[corner_idx] = [x, y, z]
                except Exception as e:
                    logger.warning(f"Error getting corner {corner_idx} for cell ({i},{j},{k}): {str(e)}")
                    raise
            
            return corners
        except Exception as e:
            logger.error(f"Error getting cell corners for ({i},{j},{k}): {str(e)}")
            raise
    
    def is_cell_active(self, grid_obj, i: int, j: int, k: int) -> bool:
        """Check if a cell is active.
        
        Args:
            grid_obj: Grid object
            i: I-index of the cell
            j: J-index of the cell
            k: K-index of the cell
            
        Returns:
            True if the cell is active, False otherwise
        """
        try:
            # First, try using the ACTNUM data if available
            if self.actnum is not None:
                try:
                    nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
                    # Calculate the global index from i,j,k
                    global_idx = i + j*nx + k*nx*ny
                    
                    # Check bounds
                    if global_idx < len(self.actnum):
                        return self.actnum[global_idx] == 1
                except Exception as e:
                    logger.warning(f"Error using ACTNUM data for cell ({i},{j},{k}): {str(e)}")
                    # Fall through to other methods
            
            # Try different methods depending on the grid API
            if hasattr(grid_obj, 'active'):
                return grid_obj.active(i, j, k)
            elif hasattr(grid_obj, 'is_active'):
                return grid_obj.is_active(i, j, k)
            else:
                # Try to get the active index - if it returns a valid index, the cell is active
                try:
                    if hasattr(grid_obj, 'get_active_index'):
                        active_idx = grid_obj.get_active_index(i, j, k)
                    elif hasattr(grid_obj, 'getActiveIndex'):
                        active_idx = grid_obj.getActiveIndex(i, j, k)
                    else:
                        # No method available to check activity
                        return True  # Assume active if we can't check
                    return active_idx >= 0
                except Exception:
                    # If this fails, assume the cell is inactive
                    return False
        except Exception as e:
            logger.error(f"Error checking if cell ({i},{j},{k}) is active: {str(e)}")
            return False

    def __del__(self):
        """Ensure restart file is closed when object is deleted."""
        self.close_restart_file() 

    def create_fault_multiplier_array(self, multflt_file, faults_file):
        """Create a 3D array of fault transmissibility multipliers.
        
        Reads fault definitions from FAULTS file and multiplier values from MULTFLT file,
        then creates a 3D property array matching the grid dimensions.
        
        Args:
            multflt_file: Path to the MULTFLT include file
            faults_file: Path to the FAULTS file
            
        Returns:
            3D numpy array of fault transmissibility multipliers with shape (nz, nx, ny)
        """
        if not all([self.nx, self.ny, self.nz]):
            logger.error("Grid dimensions not set. Call read_grid() or read_init() first.")
            return None
            
        # Initialize fault multiplier array with 1.0 (default value)
        logger.info(f"Creating fault multiplier array with dimensions: ({self.nz}, {self.nx}, {self.ny})")
        fault_array = np.ones((self.nz, self.nx, self.ny), dtype=np.float32)
        
        # Parse MULTFLT file for fault multipliers
        fault_multipliers = {}
        try:
            with open(multflt_file, 'r') as f:
                content = f.read()
                
            # Look for the MULTFLT keyword section
            if 'MULTFLT' not in content:
                logger.error(f"MULTFLT keyword not found in {multflt_file}")
                return fault_array
                
            # Parse each line in the MULTFLT section
            found_keyword = False
            for line in content.splitlines():
                line = line.strip()
                
                if not found_keyword:
                    if line == 'MULTFLT':
                        found_keyword = True
                    continue
                    
                # Skip comments, empty lines, and end marker
                if not line or line.startswith('--') or line == '/':
                    continue
                    
                # Check if we've reached the end of the section
                if line.startswith('/'):
                    break
                    
                # Parse fault multiplier: NAME VALUE /
                parts = line.split()
                if len(parts) >= 2:
                    fault_name = parts[0].strip()
                    try:
                        # Remove trailing '/' if present
                        mult_val = parts[1].strip()
                        if '/' in mult_val:
                            mult_val = mult_val.split('/')[0].strip()
                        multiplier_value = float(mult_val)
                        fault_multipliers[fault_name] = multiplier_value
                        logger.debug(f"Parsed fault {fault_name}: multiplier = {multiplier_value}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing multiplier in line: {line} - {e}")
                        
            logger.info(f"Parsed {len(fault_multipliers)} fault multipliers from {multflt_file}")
            
            # If we don't have any multipliers, return the default array
            if not fault_multipliers:
                logger.warning("No fault multipliers found. Returning array filled with 1.0.")
                return fault_array
                
            # Parse FAULTS file to get grid cell indices for each fault
            try:
                with open(faults_file, 'r') as f:
                    content = f.read()
                    
                # Look for the FAULTS keyword section
                if 'FAULTS' not in content:
                    logger.error(f"FAULTS keyword not found in {faults_file}")
                    return fault_array
                    
                # Process the FAULTS section
                found_keyword = False
                cell_count = 0
                fault_counts = {}  # To track cells per fault
                
                for line in content.splitlines():
                    line = line.strip()
                    
                    if not found_keyword:
                        if line == 'FAULTS':
                            found_keyword = True
                        continue
                    
                    # Skip comments, empty lines, and section markers
                    if not line or line.startswith('--'):
                        continue
                        
                    # Check if we've reached the end of the section
                    if line.startswith('/'):
                        break
                        
                    # Parse fault definition: 'NAME' IX1 IX2 IY1 IY2 IZ1 IZ2 FACE
                    parts = line.split()
                    if len(parts) < 8:  # Need at least 8 parts
                        continue
                        
                    # Clean up the fault name (remove quotes)
                    fault_name = parts[0].replace("'", "").strip()
                    
                    # If this fault doesn't have a multiplier, skip it
                    if fault_name not in fault_multipliers:
                        continue
                        
                    multiplier = fault_multipliers[fault_name]
                    
                    try:
                        # Eclipse uses 1-based indexing, convert to 0-based
                        ix1 = int(parts[1]) - 1
                        ix2 = int(parts[2]) - 1
                        iy1 = int(parts[3]) - 1
                        iy2 = int(parts[4]) - 1
                        iz1 = int(parts[5]) - 1
                        iz2 = int(parts[6]) - 1
                        # face = parts[7].strip()  # Not needed for now
                        
                        # Apply multiplier to all cells in the range
                        for i in range(min(ix1, ix2), max(ix1, ix2) + 1):
                            for j in range(min(iy1, iy2), max(iy1, iy2) + 1):
                                for k in range(min(iz1, iz2), max(iz1, iz2) + 1):
                                    # Ensure indices are within bounds
                                    if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                                        # Use [k, i, j] ordering for (nz, nx, ny) shape
                                        fault_array[k, i, j] = multiplier
                                        cell_count += 1
                                        
                                        # Track cells per fault
                                        if fault_name not in fault_counts:
                                            fault_counts[fault_name] = 0
                                        fault_counts[fault_name] += 1
                                    else:
                                        logger.warning(f"Cell ({i},{j},{k}) for fault {fault_name} is out of bounds.")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing fault indices in line: {line} - {e}")
                
                # Log summary of applied multipliers
                logger.info(f"Applied fault multipliers to {cell_count} cells across {len(fault_counts)} faults")
                for fault_name, count in fault_counts.items():
                    logger.info(f"  - Fault {fault_name}: multiplier {fault_multipliers[fault_name]} applied to {count} cells")
                
                return fault_array
                
            except Exception as e:
                logger.error(f"Error processing FAULTS file {faults_file}: {e}")
                return fault_array
                
        except Exception as e:
            logger.error(f"Error processing MULTFLT file {multflt_file}: {e}")
            return fault_array 

    def read_summary_rates(self, data_file: Union[str, Path], 
                         report_steps: List[int],
                         rate_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Read connection production and injection rates from UNSMRY files and map to 3D grid.
        
        This method reads cell-based connection rates from Eclipse summary files using res2df
        and maps them to the 3D grid. The rates are for oil, water, gas, and total production
        and injection.
        
        Args:
            data_file: Path to Eclipse DATA file that references the summary files.
            report_steps: List of report steps to read, should match the steps used for dynamic data.
            rate_types: List of rate types to read. Can include any of:
                ["COPR", "CWPR", "CGPR", "CWIR", "CGIR", "CVPR", "CVIR"]. If None, reads all.
                
        Returns:
            Dictionary of rates mapped to 3D grid with shape (steps, nz, nx, ny)
        """
        try:
            import res2df
            from res2df import summary, ResdataFiles
            logger.info("Successfully imported res2df packages")
        except ImportError as e:
            logger.error(f"The 'res2df' library is required but not installed: {e}")
            logger.error("Please install using: pip install res2df")
            return {}
        
        if not self.nx or not self.ny or not self.nz:
            logger.error("Grid dimensions not available. Make sure to read grid first.")
            return {}
            
        logger.info(f"Reading connection rates from {data_file} for {len(report_steps)} report steps")
        
        # Define list of rate types if not provided
        if rate_types is None:
            rate_types = ["COPR", "CWPR", "CGPR", "CWIR", "CGIR", "CVPR", "CVIR"]
        logger.info(f"Reading rate types: {rate_types}")
        
        # Initialize rate dictionary with zeros for all report steps and cells
        rates_dict = {}
        for rate_type in rate_types:
            # Initialize 4D array (time, nz, nx, ny) filled with zeros
            rates_dict[rate_type] = np.zeros((len(report_steps), self.nz, self.nx, self.ny), dtype=np.float32)
        
        try:
            # Make sure the DATA file exists
            data_file_path = Path(data_file)
            if not data_file_path.exists():
                logger.error(f"DATA file not found at path: {data_file_path}")
                return {}
                
            # Check if related SMSPEC and UNSMRY files exist in the same directory
            data_dir = data_file_path.parent
            logger.info(f"Checking for summary files in directory: {data_dir}")
            
            smspec_files = list(data_dir.glob("*.SMSPEC")) + list(data_dir.glob("*.smspec"))
            if not smspec_files:
                logger.warning(f"No SMSPEC files found in {data_dir}")
            else:
                logger.info(f"Found SMSPEC files: {[f.name for f in smspec_files]}")
                
            unsmry_files = list(data_dir.glob("*.UNSMRY")) + list(data_dir.glob("*.unsmry"))
            if not unsmry_files:
                logger.warning(f"No UNSMRY files found in {data_dir}")
            else:
                logger.info(f"Found UNSMRY files: {[f.name for f in unsmry_files]}")
            
            # Read summary data using res2df
            logger.info(f"Initializing ResdataFiles with DATA file: {data_file}")
            try:
                resdatafiles = ResdataFiles(str(data_file))
                logger.info("Successfully initialized ResdataFiles")
            except Exception as e:
                logger.error(f"Failed to initialize ResdataFiles: {e}")
                logger.info("Trying to recover by finding case name from DATA file...")
                
                # Try to extract case name and find summary files manually
                case_name = data_file_path.stem
                logger.info(f"Trying with case name: {case_name}")
                
                # Try alternative summary file reading approach
                try:
                    from resdata import summary as rd_summary
                    logger.info(f"Trying to read directly with resdata.summary using case: {case_name}")
                    summary_obj = rd_summary.Summary(str(data_dir / case_name))
                    if summary_obj:
                        logger.info("Successfully loaded summary with resdata directly")
                        # Create a pandas DataFrame equivalent to what res2df would have created
                        data = {}
                        for vector in summary_obj.keys():
                            if vector.startswith('C'):  # Get only connection vectors
                                try:
                                    data[vector] = summary_obj.numpy_vector(vector)
                                except Exception as ve:
                                    logger.warning(f"Failed to extract vector {vector}: {ve}")
                        
                        # Create DataFrame with TIME index
                        time_vector = 'TIME'
                        if time_vector in summary_obj.keys():
                            times = summary_obj.numpy_vector(time_vector)
                            dframe = pd.DataFrame(data, index=times)
                            logger.info(f"Created DataFrame with {len(dframe.columns)} vectors and {len(dframe)} time steps")
                        else:
                            # No TIME vector, use DAYS or just indices
                            if 'DAYS' in summary_obj.keys():
                                times = summary_obj.numpy_vector('DAYS')
                                dframe = pd.DataFrame(data, index=times)
                            else:
                                dframe = pd.DataFrame(data)
                                
                        logger.info(f"Created summary DataFrame with shape: {dframe.shape}")
                    else:
                        logger.error(f"Failed to load summary with resdata directly")
                        return {}
                except Exception as rd_error:
                    logger.error(f"Failed to read summary with resdata directly: {rd_error}")
                    return {}
            else:
                # Original approach continues here if ResdataFiles initialization succeeded
                try:
                    # Get connection data (C* vectors) as a pandas DataFrame
                    logger.info("Calling res2df.summary.df to get summary data")
                    dframe = summary.df(resdatafiles, column_keys="C*", time_index="raw")
                    logger.info(f"Read summary dataframe with shape: {dframe.shape}")
                    
                    # Check if we got any data
                    if dframe.empty:
                        logger.warning("Summary DataFrame is empty - no connection vectors found")
                        # Try getting all vectors to see what's available
                        try:
                            all_df = summary.df(resdatafiles, time_index="raw")
                            logger.info(f"Full summary dataframe has shape: {all_df.shape}")
                            logger.info(f"Available column patterns: {sorted(set([col.split(':')[0] for col in all_df.columns if ':' in col]))}")
                            # Try another pattern if C* didn't work
                            dframe = all_df.filter(regex='^C.*:')
                            logger.info(f"Using regex filter, got DataFrame with shape: {dframe.shape}")
                        except Exception as all_err:
                            logger.warning(f"Error getting all vectors: {all_err}")
                except Exception as df_error:
                    logger.error(f"Failed to create summary DataFrame: {df_error}")
                    return {}
            
            # Check if we have data
            if dframe.empty:
                logger.warning("No connection rate data found in summary files")
                return {}
                
            # First extract all times from the dataframe
            times = dframe.index.values
            logger.debug(f"Available timesteps in summary data: {len(times)}")
            
            # Extract the times corresponding to the requested report steps
            if self.rst_file and hasattr(self.rst_file, 'report_dates'):
                # If we have report dates, try to match with summary timestamps
                report_dates = self.rst_file.report_dates
                logger.debug(f"Matching with report dates from UNRST file")
                
                # Debug report dates from UNRST
                if report_dates:
                    if len(report_dates) > 3:
                        logger.info(f"UNRST report dates (first 3): {report_dates[:3]}")
                    else:
                        logger.info(f"UNRST report dates: {report_dates}")
                
                # Debug summary times
                if len(times) > 3:
                    logger.info(f"Summary times (first 3): {times[:3]}")
                else:
                    logger.info(f"Summary times: {times}")
                
                # Use a more robust matching approach
                time_indices = []
                
                # Check if times and report_dates are comparable
                if len(times) > 0 and hasattr(times[0], 'date'):
                    # Convert all summary times to dates for easier comparison
                    summary_dates = [t.date() if hasattr(t, 'date') else t for t in times]
                    
                    # For each report step, find the closest matching date
                    for step_idx, step in enumerate(report_steps):
                        if step_idx < len(self.available_steps) and self.available_steps[step_idx] == step:
                            # Use report step index directly if it matches
                            date_idx = step_idx
                        else:
                            # Find the index of the step in available_steps
                            try:
                                date_idx = self.available_steps.index(step)
                            except ValueError:
                                logger.warning(f"Report step {step} not in available_steps {self.available_steps}")
                                # Use a fallback approach - take step_idx or nearest available
                                if step_idx < len(self.available_steps):
                                    date_idx = step_idx
                                else:
                                    logger.warning(f"Cannot find date index for step {step}, skipping")
                                    continue
                        
                        # Get the report date for this step if available
                        if date_idx < len(report_dates):
                            target_date = report_dates[date_idx]
                            target_date_only = target_date.date() if hasattr(target_date, 'date') else target_date
                            
                            # Find closest date in summary timestamps
                            closest_idx = None
                            min_diff = float('inf')
                            
                            for i, date in enumerate(summary_dates):
                                # Simplified date comparison - just look for exact match or nearest
                                if date == target_date_only:
                                    closest_idx = i
                                    break
                                
                                # If no exact match, try to find nearest by simple comparison
                                try:
                                    # Use simple ordering comparison which works for dates
                                    if date < target_date_only:
                                        diff = (target_date_only - date).days
                                    else:
                                        diff = (date - target_date_only).days
                                        
                                    if diff < min_diff:
                                        min_diff = diff
                                        closest_idx = i
                                except (TypeError, AttributeError):
                                    # If dates can't be compared, skip
                                    pass
                            
                            if closest_idx is not None:
                                time_indices.append(closest_idx)
                                logger.info(f"Matched report step {step} to summary timestamp index {closest_idx} (diff={min_diff} days)")
                            else:
                                logger.warning(f"Could not find matching timestamp for report step {step}")
                                # Fallback: use a proportional matching based on position
                                fallback_idx = min(int(step_idx * len(times) / len(report_steps)), len(times) - 1)
                                time_indices.append(fallback_idx)
                                logger.warning(f"Using fallback position-based index {fallback_idx} for step {step}")
                        else:
                            logger.warning(f"Report step {step} (idx {date_idx}) has no date in report_dates (length {len(report_dates)})")
                            # Fallback: use a proportional matching based on position
                            fallback_idx = min(int(step_idx * len(times) / len(report_steps)), len(times) - 1)
                            time_indices.append(fallback_idx)
                            logger.warning(f"Using fallback position-based index {fallback_idx} for step {step}")
                
                # If we couldn't match any steps, fall back to simpler approach
                if not time_indices:
                    logger.warning(f"Could not match any report steps with timestamps. Using position-based fallback.")
                    # Just use equally spaced indices across the available times
                    time_indices = [min(int(i * len(times) / len(report_steps)), len(times) - 1) for i in range(len(report_steps))]
                    logger.info(f"Using fallback time indices: {time_indices}")
            else:
                # If no report dates available, use equally spaced indices
                logger.info("No report dates available. Using position-based indices.")
                # Distribute requested steps evenly across available timesteps
                time_indices = [min(int(i * len(times) / len(report_steps)), len(times) - 1) for i in range(len(report_steps))]
                logger.info(f"Using position-based time indices: {time_indices}")
                
            # Ensure we have time indices for all report steps
            if len(time_indices) < len(report_steps):
                logger.warning(f"Not enough time indices ({len(time_indices)}) for all report steps ({len(report_steps)})")
                # Pad with defaults
                needed = len(report_steps) - len(time_indices)
                padding = [len(times) - 1] * needed
                time_indices.extend(padding)
                logger.info(f"Padded time indices to length {len(time_indices)}")
                
            # Log the final time mapping
            time_mapping = [f"{step} -> {times[idx] if idx < len(times) else 'N/A'}" 
                          for step, idx in zip(report_steps, time_indices) if idx < len(times)]
            logger.info(f"Time mapping (first few): {time_mapping[:min(5, len(time_mapping))]}")
            
            # Log the columns we found
            logger.info(f"Found {len(dframe.columns)} columns in summary data")
            if len(dframe.columns) < 10:
                logger.info(f"All columns: {list(dframe.columns)}")
            else:
                logger.info(f"First 10 columns: {list(dframe.columns)[:10]}")
                
            # Process each rate type
            for rate_type in rate_types:
                logger.info(f"Processing rate type: {rate_type}")
                
                # Filter columns for current rate type
                rate_columns = dframe.filter(like=f"{rate_type}:").columns
                logger.info(f"Found {len(rate_columns)} columns for {rate_type}")
                
                if len(rate_columns) == 0:  # Fix: use len() instead of direct boolean check
                    logger.warning(f"No columns found for rate type {rate_type}")
                    continue
                
                # Process each connection
                cell_count = 0
                
                # Add counters for message limiting
                no_cell_indices_count = 0
                no_cell_indices_limit = 20
                out_of_bounds_count = 0
                out_of_bounds_limit = 20
                parse_error_count = 0
                parse_error_limit = 20
                
                for column in rate_columns:
                    try:
                        # Parse the column name to extract cell indices
                        # Format: "COPR:WELL_NAME:i,j,k" or sometimes with an integer instead of i,j,k
                        parts = column.split(':')
                        if len(parts) < 3:
                            logger.debug(f"Skipping column {column} - doesn't match expected format")
                            continue
                            
                        # Check if the last part contains cell indices (i,j,k)
                        cell_indices = parts[-1]
                        if ',' in cell_indices:
                            # Extract i,j,k values
                            try:
                                i, j, k = map(int, cell_indices.split(','))
                                # Adjust for 1-based indexing in Eclipse
                                i -= 1
                                j -= 1 
                                k -= 1
                                
                                # Check if indices are within grid bounds
                                if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                                    # Get rate values for the selected time steps
                                    for step_idx, time_idx in enumerate(time_indices):
                                        if time_idx < len(times) and step_idx < len(report_steps):
                                            time = times[time_idx]
                                            if time in dframe.index:
                                                # Map the rate to the 3D grid
                                                rate_value = dframe.loc[time, column]
                                                # Store in 4D array (time, z, x, y)
                                                rates_dict[rate_type][step_idx, k, i, j] = rate_value
                                                cell_count += 1
                                            else:
                                                logger.warning(f"Time {time} not in dataframe index")
                                else:
                                    # Log out-of-bounds message with limiting
                                    if out_of_bounds_count < out_of_bounds_limit:
                                        logger.debug(f"Cell indices ({i},{j},{k}) out of grid bounds ({self.nx},{self.ny},{self.nz})")
                                    elif out_of_bounds_count == out_of_bounds_limit:
                                        logger.debug(f"Suppressing further out-of-bounds messages...")
                                    out_of_bounds_count += 1
                            except ValueError:
                                # Log parse error with limiting
                                if parse_error_count < parse_error_limit:
                                    logger.debug(f"Could not parse cell indices from {cell_indices}")
                                elif parse_error_count == parse_error_limit:
                                    logger.debug(f"Suppressing further cell indices parse error messages...")
                                parse_error_count += 1
                        else:
                            # Log no cell indices message with limiting
                            if no_cell_indices_count < no_cell_indices_limit:
                                logger.debug(f"No cell indices in {column}, format incompatible")
                            elif no_cell_indices_count == no_cell_indices_limit:
                                logger.debug(f"Suppressing further 'No cell indices' messages...")
                            no_cell_indices_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error processing column {column}: {str(e)}")
                
                # Log summary of message limits
                if no_cell_indices_count > no_cell_indices_limit:
                    logger.debug(f"Suppressed {no_cell_indices_count - no_cell_indices_limit} 'No cell indices' messages")
                if out_of_bounds_count > out_of_bounds_limit:
                    logger.debug(f"Suppressed {out_of_bounds_count - out_of_bounds_limit} out-of-bounds messages")
                if parse_error_count > parse_error_limit:
                    logger.debug(f"Suppressed {parse_error_count - parse_error_limit} cell indices parse error messages")
                
                logger.info(f"Processed {cell_count} cell entries for rate type {rate_type}")
            
            # Calculate total production rate Q = Qo + Qw + Qg
            if all(key in rates_dict for key in ["COPR", "CWPR", "CGPR"]):
                logger.info("Calculating total production rate (Q = Qo + Qw + Qg)")
                # Initialize with same shape as other rate arrays
                total_rate = np.zeros_like(rates_dict["COPR"])
                # Add oil, water, and gas production rates
                total_rate += rates_dict["COPR"]  # Qo
                total_rate += rates_dict["CWPR"]  # Qw
                total_rate += rates_dict["CGPR"]  # Qg
                # Add to the dictionary
                rates_dict["Q"] = total_rate
                logger.info(f"Added total production rate Q with shape {total_rate.shape}")
                
                # Check if we have non-zero values
                non_zero_cells = np.count_nonzero(total_rate)
                logger.info(f"Total production rate has {non_zero_cells} non-zero cell values")
            else:
                logger.warning("Cannot calculate total production rate: missing one or more required rate types")
                
            logger.info(f"Successfully processed {len(rate_types)} rate types")
            return rates_dict
            
        except Exception as e:
            logger.error(f"Error reading summary rates: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def extract_well_production_data(
        self, 
        data_file: Union[str, Path], 
        report_steps: List[int], 
        requested_steps: List[int], 
        well_types: Dict[str, str] = None,
        producer_only: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract well production data from summary files for specific wells.
        
        Args:
            data_file: Path to Eclipse DATA file
            report_steps: All available report steps
            requested_steps: Specific report steps to extract data for
            well_types: Dictionary mapping well names to types (from read_well_types_from_schedule)
            producer_only: If True, only extract data for producer wells
            
        Returns:
            Dictionary mapping well names to production metrics
        """
        try:
            import res2df
            from res2df import summary, ResdataFiles
            logger.info("Successfully imported res2df packages")
        except ImportError as e:
            logger.error(f"The 'res2df' library is required but not installed: {e}")
            logger.error("Please install using: pip install res2df")
            return {}
            
        logger.info(f"Extracting well production data from {data_file}")
        
        # Skip special metadata entries
        special_keys = ['__original_names__', '__normalized_types__']
        
        # Create mapping of normalized names to original names if available
        name_mapping = {}
        original_names = well_types.get('__original_names__', {}) if well_types else {}
        normalized_types = well_types.get('__normalized_types__', {}) if well_types else {}
        
        # Get wells to extract based on well_types
        wells_to_extract = []
        if well_types:
            if producer_only:
                wells_to_extract = [w for w, t in well_types.items() 
                                 if t == 'producer' and w not in special_keys]
            else:
                wells_to_extract = [w for w in well_types.keys() 
                                 if w not in special_keys]
        
        if not wells_to_extract:
            logger.warning("No wells specified for extraction")
            return {}
            
        logger.info(f"Extracting production data for {len(wells_to_extract)} wells")
        
        # Add both original and normalized well names to check
        wells_to_check = set()
        for well in wells_to_extract:
            wells_to_check.add(well)  # Original name
            # Add normalized version
            normalized = normalize_well_name(well)
            wells_to_check.add(normalized)
            # Track mapping
            name_mapping[normalized] = well
        
        # Add original names from well_types if available
        if original_names:
            for norm_name, orig_name in original_names.items():
                if orig_name in wells_to_extract:
                    wells_to_check.add(norm_name)
                    name_mapping[norm_name] = orig_name
        
        logger.info(f"Looking for production data for {len(wells_to_extract)} wells "
                   f"({len(wells_to_check)} including normalized variants)")
        
        # Define the production metrics to extract
        production_metrics = {
            'WOPR': 'oil_rate',          # Well Oil Production Rate
            'WWPR': 'water_rate',        # Well Water Production Rate
            'WGPR': 'gas_rate',          # Well Gas Production Rate
            'WOPT': 'oil_total',         # Well Oil Production Total (cumulative)
            'WGOR': 'gas_oil_ratio',     # Well Gas-Oil Ratio
            'WWCT': 'water_cut'          # Well Water Cut
        }
        
        production_keys = [f"{key}:*" for key in production_metrics.keys()]
        
        try:
            # Initialize ResdataFiles
            resdatafiles = ResdataFiles(str(data_file))
            
            # Get production data as DataFrame
            production_df = summary.df(resdatafiles, column_keys=production_keys, time_index="raw")
            
            if production_df.empty:
                logger.warning("No production data found in summary file")
                return {}
                
            # Log production data details
            prod_cols = list(production_df.columns)
            logger.info(f"Production data has {len(prod_cols)} columns and {len(production_df)} rows")
            logger.debug(f"First few production columns: {prod_cols[:min(10, len(prod_cols))]}")
            
            # Extract available steps
            available_steps = list(range(len(production_df)))
            logger.info(f"Production data has {len(available_steps)} timesteps")
            
            # Log well names found in production data
            prod_wells = set()
            for col in prod_cols:
                parts = col.split(':')
                if len(parts) >= 2:
                    prod_wells.add(parts[1])
            logger.info(f"Found {len(prod_wells)} wells in production data")
            logger.info(f"Wells in production data: {', '.join(sorted(list(prod_wells)))}")
            
            # Filter columns to only include those with production metrics
            well_found = set()  # Track which wells we found data for
            
            for col in prod_cols:
                col_parts = col.split(':')
                if len(col_parts) >= 2:
                    metric = col_parts[0]
                    well = col_parts[1]
                    
                    # Check with original and normalized well names
                    if metric in production_metrics:
                        normalized_well = normalize_well_name(well)
                        
                        # Check both original and normalized forms
                        if well in wells_to_check or normalized_well in wells_to_check:
                            # Record which well this matches
                            if well in wells_to_extract:
                                well_found.add(well)
                            elif normalized_well in name_mapping:
                                # This is a normalized name that matches one of our target wells
                                well_found.add(name_mapping[normalized_well])
            
            logger.info(f"Found production data for {len(well_found)} wells")
            logger.info(f"Wells with production data: {', '.join(sorted(list(well_found)))}")
            
            # Extract data for each well
            production_data = {}
            
            for well in wells_to_extract:
                if well not in well_found:
                    continue  # Skip wells with no data
                    
                well_metrics = {}
                
                # Initialize with empty data for each metric
                for metric_key, metric_name in production_metrics.items():
                    well_metrics[metric_name] = np.zeros(len(requested_steps), dtype=np.float32)
                
                # Also check normalized version of well name
                normalized_well = normalize_well_name(well)
                
                # Extract each metric for this well
                for metric_key, metric_name in production_metrics.items():
                    # Try both original and normalized well names
                    column_found = False
                    value_found = False
                    
                    # First try original well name
                    col_name = f"{metric_key}:{well}"
                    if col_name in production_df.columns:
                        column_found = True
                        
                        # Extract values for each report step
                        for i, step in enumerate(requested_steps):
                            if step in available_steps:
                                step_idx = available_steps.index(step)
                                # Get the value at this step if available
                                if step_idx < len(production_df):
                                    value = production_df.iloc[step_idx][col_name]
                                    if not pd.isna(value):
                                        well_metrics[metric_name][i] = value
                                        value_found = True
                    
                    # If not found or no values, try with normalized well name
                    if not column_found or not value_found:
                        # Try alternate column name formats
                        potential_cols = []
                        
                        # Check for columns with this metric that contain the well name
                        for col in production_df.columns:
                            if col.startswith(f"{metric_key}:"):
                                col_well = col.split(':')[1]
                                # Check if this is our well with different formatting
                                if normalize_well_name(col_well) == normalized_well:
                                    potential_cols.append(col)
                        
                        if potential_cols:
                            for alt_col in potential_cols:
                                # Extract values for each report step
                                for i, step in enumerate(requested_steps):
                                    if step in available_steps:
                                        step_idx = available_steps.index(step)
                                        # Get the value at this step if available
                                        if step_idx < len(production_df):
                                            value = production_df.iloc[step_idx][alt_col]
                                            if not pd.isna(value):
                                                well_metrics[metric_name][i] = value
                                                value_found = True
                                
                                # If we found values with this column, don't try others
                                if value_found:
                                    break
                
                # Only add well to results if it has some non-zero data
                if any(np.any(metrics) for metrics in well_metrics.values()):
                    production_data[well] = well_metrics
            
            logger.info(f"Extracted production data for {len(production_data)} wells")
            return production_data
            
        except Exception as e:
            logger.error(f"Error extracting well production data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}