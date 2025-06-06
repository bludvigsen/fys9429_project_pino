"""Script to prepare training data from Eclipse files for PINO surrogate model.

This script loads data from Eclipse simulation files and prepares it in the format
expected by the Forward_problem.py script for training a Fourier neural operator.
It handles multiple simulations organized in realization directories.

@Author: Bjørn Egil Ludvigsen, Aker BP
"""

import logging
import os
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
logger = logging.getLogger(__name__)

import numpy as np
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import sys

# Add the parent directory to the path so imports work correctly
# regardless of where the script is run from
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
    
# Now we can import from core
try:
    from core.eclipse_reader import EclipseReader, normalize_well_name, map_active_to_full_grid
except ImportError:
    try:
        from .eclipse_reader import EclipseReader, normalize_well_name, map_active_to_full_grid
    except ImportError:
        from eclipse_reader import EclipseReader # map_active_to_full_grid needs to be handled here if standalone
        
        # Define normalize_well_name if import fails
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

try:
    import resdata.resfile # Import the resdata library
except ImportError:
    logger.error("The 'resdata' library is required but not installed. Please install it (e.g., pip install resdata).")
    sys.exit(1) # Exit if resdata is crucial and not found



def load_actnum(file_path: Union[str, Path]) -> Optional[np.ndarray]: # Return Optional for critical failure
    """Load ACTNUM map from grdecl file, handling the multiplier format.
        
    Args:
        file_path: Path to ACTNUM.grdecl file
            
    Returns:
        ACTNUM array, or None if critical parsing error occurs.
    """
    file_path = Path(file_path)
    logger.debug(f"Attempting to load ACTNUM from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        actnum_data_started = False
        values = []
        
        for line_num, line_content in enumerate(lines):
            line = line_content.strip()

            if not actnum_data_started:
                # Case-insensitive match for the keyword, ensuring it's the whole word
                # and not part of another keyword.
                # Splitting the line and checking the first token might be more robust
                # if ACTNUM can be followed by data on the same line.
                # For now, assume ACTNUM is on its own line or the first keyword on a line.
                if line.upper().startswith('ACTNUM'):
                    # Check if it's exactly 'ACTNUM' or 'ACTNUM ' to avoid matching e.g. 'ACTNUMS'
                    tokens = line.upper().split()
                    if tokens and tokens[0] == 'ACTNUM':
                        actnum_data_started = True
                        logger.debug(f"Found ACTNUM keyword at line {line_num + 1} in {file_path}")
                        # If ACTNUM is followed by data on the same line, process it here.
                        # This part needs careful handling based on expected format.
                        # Assuming for now data starts on the next line or after the keyword on this line.
                        # Example: if line is "ACTNUM 10*1 20*0", split further.
                        # For simplicity, current logic assumes data begins on subsequent lines
                        # or after this keyword if this line has more tokens.
                        # If ACTNUM itself is the only content, the next loop iteration will pick up data.
                        # If data can follow "ACTNUM" on the same line, the logic below needs to handle `line.split()[1:]`
                        # if tokens[0] == 'ACTNUM' and len(tokens) > 1.
                        # For now, we strictly expect data on lines *after* the ACTNUM keyword line,
                        # or that the item processing loop handles the remainder of the current line if ACTNUM is first.
                        # The current loop structure is: find ACTNUM, then in *next* iterations, process lines.
                        # This is fine if ACTNUM is a standalone header.
                        if len(tokens) == 1: # ACTNUM is the only thing on this line
                             continue # Move to next line for data
                        else: # Data might be on the same line after ACTNUM
                             line_items_for_processing = line.split()[1:] # Get items after "ACTNUM"
                             # Fall through to item processing for these items
                             pass # Let the item processing loop handle items_after_keyword
                    else: # Found "ACTNUM" as a substring but not the keyword itself
                        continue
                else:
                    continue # Skip lines until ACTNUM keyword is found

            # If we have started reading ACTNUM data (and ACTNUM wasn't the only thing on its line)
            # Or if this is a line after ACTNUM was found.
            
            # If line_items_for_processing was not set (i.e. ACTNUM was standalone or data is on new line)
            if 'line_items_for_processing' not in locals() or not line_items_for_processing:
                line_items_for_processing = line.split()

            if not line and not actnum_data_started: # Still searching for ACTNUM and line is empty
                continue
            
            if actnum_data_started: # Only process if keyword has been found
                if not line or line.startswith('--'): # Skip empty lines and comments within data block
                    if 'line_items_for_processing' in locals(): del line_items_for_processing # reset for next line
                    continue
                
                if line == '/': # End of ACTNUM section
                    logger.debug(f"Found end of ACTNUM section at line {line_num + 1} in {file_path}")
                    break
            
                # Process each item in the line
                for item_idx, item in enumerate(line_items_for_processing):
                    if not item: continue # Skip empty items

                    if '*' in item:
                        parts = item.split('*')
                        if len(parts) == 2:
                            try:
                                count = int(parts[0])
                                value = int(parts[1])
                                if count < 0:
                                    logger.warning(f"Negative count '{count}' in multiplier '{item}' in {file_path} at line {line_num + 1}, item {item_idx + 1}. Interpreting as 0 count.")
                                    count = 0
                                values.extend([value] * count)
                            except ValueError:
                                logger.error(f"Invalid number in multiplier '{item}' in {file_path} at line {line_num + 1}, item {item_idx + 1}. Skipping item.")
                        else:
                            logger.warning(f"Malformed multiplier '{item}' (expected count*value) in {file_path} at line {line_num + 1}, item {item_idx + 1}. Skipping item.")
                    else:
                        try:
                            values.append(int(item))
                        except ValueError:
                            logger.error(f"Invalid integer value '{item}' in {file_path} at line {line_num + 1}, item {item_idx + 1}. Skipping item.")
                
                if 'line_items_for_processing' in locals(): del line_items_for_processing # reset for next line processing
        
        if not actnum_data_started:
            logger.error(f"ACTNUM keyword not found in file: {file_path}")
            return None

        if not values: # If ACTNUM was found but no data followed before '/' or EOF
            logger.warning(f"No values found in ACTNUM section of {file_path} after keyword or section was empty.")
            return np.array([], dtype=int) 

        logger.info(f"Successfully loaded {len(values)} values from ACTNUM file: {file_path}")
        return np.array(values)
            
    except FileNotFoundError:
        logger.error(f"ACTNUM file not found: {file_path}")
        return None # Critical error, file doesn't exist
    except Exception as e:
        logger.error(f"Generic error loading ACTNUM from {file_path}: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None # Or re-raise if it should halt execution

def load_faults(fault_file: Union[str, Path], mult_file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load fault definitions and multipliers.
        
        Args:
        fault_file: Path to grid.faults file
        mult_file: Path to multflt.inc file
            
        Returns:
        Tuple of (fault array, multiplier array)
    """
    try:
        # Load fault definitions
        with open(fault_file, 'r') as f:
            fault_lines = f.readlines()
        
        # Parse fault definitions
        faults = []
        for line in fault_lines:
            if line.strip() and not line.strip().startswith('--'):
                parts = line.split()
                if len(parts) >= 6:  # Expect at least 6 values for fault definition
                    faults.append([float(x) for x in parts[:6]])
                
        fault_array = np.array(faults)
    
        # Load multipliers
        with open(mult_file, 'r') as f:
            mult_lines = f.readlines()
        
        # Parse multipliers
        multipliers = []
        for line in mult_lines:
            if line.strip() and not line.strip().startswith('--'):
                parts = line.split()
                if len(parts) >= 2:  # Expect at least 2 values per multiplier
                    multipliers.append([float(x) for x in parts[:2]])
                
        mult_array = np.array(multipliers)
    
        return fault_array, mult_array
            
    except Exception as e:
        logger.error(f"Error loading faults: {str(e)}")
        raise

def clean_dict_arrays(
    data_dict: Dict[str, np.ndarray],
    fields_to_clip_negative: Optional[List[str]] = None,
    field_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """Clean arrays in dictionary by replacing invalid values and applying field-specific thresholds.
    
    Args:
        data_dict: Dictionary containing numpy arrays
        fields_to_clip_negative: Optional list of field names where negative values should be set to 0.
            If None, uses default fields (input, output, and flow fields)
        field_thresholds: Optional dictionary mapping field names to (min, max) threshold tuples.
            Example: {'pressure': (0, 500), 'porosity': (0, 1)}
            If None or field not specified, uses float32 min/max as defaults
    
    Returns:
        Dictionary with cleaned arrays
    """
    # Define default threshold as the largest finite representable number for np.float32
    default_max = np.finfo(np.float32).max
    default_min = np.finfo(np.float32).min
    
    # Default fields to clip negative values if none provided
    if fields_to_clip_negative is None:
        fields_to_clip_negative = [
            'permeability', 'porosity', 'Fault', 'Pini', 'Sini',
            'Pressure', 'Water_saturation', 'Gas_saturation', 'Oil_saturation',
            'Q', 'Qw', 'Qg', 'Qo'
        ]
    
    # Initialize field_thresholds if None
    if field_thresholds is None:
        field_thresholds = {}
    
    # Create a new dictionary for the cleaned data
    cleaned_dict = {}
    
    for key, array in data_dict.items():
        # Skip non-array items or copy them directly
        if not isinstance(array, np.ndarray):
            cleaned_dict[key] = array
            continue
            
        # Skip arrays with non-numeric types (like strings or booleans)
        if array.dtype.kind not in 'fcui':  # float, complex, unsigned int, int
            cleaned_dict[key] = array
            continue
        
        # For integer arrays, just copy them (grid dimensions, indices, etc.)
        if array.dtype.kind in 'ui':  # unsigned int, int
            # Only check for negative values if applicable and requested
            if key in fields_to_clip_negative and array.dtype.kind == 'i':
                array = np.maximum(array, 0)
            cleaned_dict[key] = array
            continue
        
        # Now we're dealing with float or complex arrays
        # Make a copy to avoid modifying the original
        cleaned_array = array.copy()
        
        # Get field-specific thresholds or use defaults
        min_threshold, max_threshold = field_thresholds.get(key, (default_min, default_max))
        
        # Replace NaNs, infinities and values outside thresholds
        try:
            invalid_indices = (~np.isfinite(cleaned_array)) | \
                             (cleaned_array < min_threshold) | \
                             (cleaned_array > max_threshold)
            cleaned_array[invalid_indices] = 0.0
            
            # Clip negative values for specified fields
            if key in fields_to_clip_negative:
                cleaned_array = np.clip(cleaned_array, 0, None)
            
            # Apply field-specific thresholds
            cleaned_array = np.clip(cleaned_array, min_threshold, max_threshold)
            
        except TypeError as e:
            logger.warning(f"Could not clean array '{key}' with dtype {array.dtype}: {e}")
            # If cleaning fails, just use the original array
            
        # Convert to float32
        if cleaned_array.dtype != np.float32:
            try:
                cleaned_array = cleaned_array.astype(np.float32)
            except Exception as e:
                logger.warning(f"Could not convert '{key}' to float32: {e}")
        
        cleaned_dict[key] = cleaned_array
    
    return cleaned_dict

def load_grid_data(filepath: Union[str, Path]) -> object:
    """Load grid data from EGRID file.

    Args:
        filepath: Path to EGRID file

    Returns:
        Grid object from EclipseReader
    """
    logger.info(f"Loading grid data from: {filepath}")
    # Instantiate EclipseReader locally just for grid reading
    reader = EclipseReader(filepath)
    grid_obj = reader.read_grid()
    if grid_obj:
        logger.info(f"Grid dimensions: {grid_obj.nx}x{grid_obj.ny}x{grid_obj.nz}")
    else:
        logger.error(f"Failed to read grid object from {filepath}")
    return grid_obj

def process_single_realization(
    realization_dir: Union[str, Path],
    nx: int,
    ny: int,
    nz: int,
    steppi: int,
    steppi_indices: List[int],
    include_rates: bool = True,
    rate_types: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """Process data from a single realization using enhanced EclipseReader.

    Args:
        realization_dir: Path to realization directory
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        nz: Number of cells in z direction
        steppi: Number of timesteps
        steppi_indices: List of report steps to include from UNRST file
        include_rates: Whether to include production/injection rates
        rate_types: Optional list of rate types to include. Default includes all.
            
    Returns:
        Dictionary containing simulation data
    """
    realization_dir = Path(realization_dir)
    logger.info(f"Processing realization: {realization_dir.name}")
    
    # Find required files within the realization directory
    grid_file = list(realization_dir.glob("*.EGRID"))
    if not grid_file:
        raise ValueError(f"No EGRID file found in {realization_dir}")
    grid_file = grid_file[0]

    init_file = list(realization_dir.glob("*.INIT"))
    if not init_file:
        raise ValueError(f"No INIT file found in {realization_dir}")
    init_file = init_file[0]

    restart_file = list(realization_dir.glob("*.UNRST"))
    if not restart_file:
        raise ValueError(f"No UNRST file found in {realization_dir}")
    restart_file = restart_file[0]
    
    # Find DATA file for summary data (if including rates)
    data_file = None
    if include_rates:
        logger.info(f"Attempting to read production and injection rates for realization: {realization_dir.name}")
        data_files = list(realization_dir.glob("*.DATA"))
        if not data_files:
            # Try case-insensitive search
            data_files = list(realization_dir.glob("*.data"))
            
        if not data_files:
            logger.warning(f"No DATA file found in {realization_dir} for reading rates. Rates will be excluded.")
            include_rates = False
        else:
            data_file = data_files[0]
            logger.info(f"Found DATA file for rates: {data_file}")
            
        # Also check for summary files explicitly
        smspec_files = list(realization_dir.glob("*.SMSPEC"))
        if not smspec_files:
            smspec_files = list(realization_dir.glob("*.smspec"))
        if not smspec_files:
            logger.warning(f"No SMSPEC files found in {realization_dir}. Rate data may not be available.")
        else:
            logger.info(f"Found SMSPEC file: {smspec_files[0]}")
            
        unsmry_files = list(realization_dir.glob("*.UNSMRY"))
        if not unsmry_files:
            unsmry_files = list(realization_dir.glob("*.unsmry"))
        if not unsmry_files:
            logger.warning(f"No UNSMRY files found in {realization_dir}. Rate data may not be available.")
        else:
            logger.info(f"Found UNSMRY file: {unsmry_files[0]}")
    else:
        logger.info("Rate reading is disabled by include_rates=False parameter.")
    
    # Look for include directory with ACTNUM, faults, etc.
    include_dir = realization_dir / "include"
    if not include_dir.exists():
        include_dir = realization_dir
        
    actnum_file = include_dir / "ACTNUM.grdecl"
    fault_file = include_dir / "grid.faults"
    mult_file = include_dir / "multflt.inc"
    
    if not actnum_file.exists():
        actnum_files = list(include_dir.glob("*ACTNUM*.grdecl"))
        if actnum_files:
            actnum_file = actnum_files[0]
        else:
            raise ValueError(f"No ACTNUM file found in {include_dir}")
            
    if not fault_file.exists():
        fault_files = list(include_dir.glob("*fault*.faults"))
        if fault_files:
            fault_file = fault_files[0]
        else:
            logger.warning(f"No fault file found in {include_dir}. Using empty fault data.")
            fault_array = np.zeros((0, 6))
            mult_array = np.zeros((0, 2))
    
    if not mult_file.exists():
        mult_files = list(include_dir.glob("*multflt*.inc"))
        if not mult_files:
            mult_files = list(include_dir.glob("*multflt*.in"))
        if mult_files:
            mult_file = mult_files[0]
            logger.info(f"Found multiplier file: {mult_file}")
        else:
            logger.warning(f"No multiplier file found in {include_dir}. Using empty multiplier data.")
            mult_array = np.zeros((0, 2))
    
    # Load grid data using the utility function
    logger.info(f"Loading grid data from {grid_file}")
    grid_obj = load_grid_data(grid_file)
    if grid_obj is None:
        raise ValueError(f"Failed to load grid data from {grid_file}")
    
    # Load ACTNUM using the utility function
    logger.info(f"Loading ACTNUM from {actnum_file}")
    try:
        actnum_flat = load_actnum(actnum_file)
        if actnum_flat is None: # Check if load_actnum returned None (critical error)
            logger.error(f"Failed to load ACTNUM data from {actnum_file}. Cannot proceed with realization {realization_dir.name}.")
            raise ValueError(f"ACTNUM data could not be loaded for {actnum_file}")

        # Check if total number of elements matches grid dimensions
        expected_total_cells = nx * ny * nz
        if actnum_flat.size != expected_total_cells:
            logger.error(f"ACTNUM data size mismatch for {actnum_file} in realization {realization_dir.name}: "
                         f"Expected {expected_total_cells} cells (for {nx}x{ny}x{nz} grid), "
                         f"but ACTNUM file provided {actnum_flat.size} values. "
                         "Reshaping will fail. Please check ACTNUM.grdecl file and grid dimensions.")
            raise ValueError(f"ACTNUM data size {actnum_flat.size} does not match grid dimensions {nx}x{ny}x{nz}")

        # Reshape assuming Fortran order, then transpose to (nz, nx, ny)
        actnum = np.transpose(actnum_flat.reshape((nx, ny, nz), order='F'), (2, 0, 1))
        logger.info(f"Loaded and reshaped ACTNUM from grdecl file (shape: {actnum.shape}) for realization {realization_dir.name}.")

        # --- Active Cell Count Comparison ---
        if grid_obj and actnum is not None: # actnum here is the 3D reshaped array
            sum_from_actnum_file = np.sum(actnum == 1) # Count of cells marked as 1
            
            grid_obj_active_count = -1 # Initialize to indicate not found/failed
            if hasattr(grid_obj, 'get_num_active'): 
                try:
                    grid_obj_active_count = grid_obj.get_num_active()
                    logger.debug(f"[Realization: {realization_dir.name}] Successfully called grid_obj.get_num_active().")
                except Exception as e:
                    logger.error(f"[Realization: {realization_dir.name}] Error calling grid_obj.get_num_active(): {e}. "
                                 "Active cell count from grid object will not be available for comparison.")
            else:
                logger.warning(f"[Realization: {realization_dir.name}] Grid object does not have 'get_num_active' method. "
                               "Active cell count from grid object will not be available for comparison.")
            
            logger.info(f"[Realization: {realization_dir.name}] Active cells from ACTNUM.grdecl file (sum of 1s): {sum_from_actnum_file}")
            if grid_obj_active_count != -1: # Proceed with comparison only if count was successfully retrieved
                logger.info(f"[Realization: {realization_dir.name}] Active cells from grid object via get_num_active(): {grid_obj_active_count}")
                if sum_from_actnum_file != grid_obj_active_count:
                    logger.error(f"[Realization: {realization_dir.name}] MISMATCH in active cell count: "
                                 f"ACTNUM.grdecl indicates {sum_from_actnum_file} active cells, "
                                 f"while grid object (get_num_active) reports {grid_obj_active_count} active cells. "
                                 f"This may indicate inconsistencies in grid definitions or ACTNUM interpretation.")
                else:
                    logger.info(f"[Realization: {realization_dir.name}] Active cell count MATCHES between ACTNUM.grdecl and grid object ({sum_from_actnum_file}).")
            else:
                # This block is reached if get_num_active was not available or failed
                logger.warning(f"[Realization: {realization_dir.name}] Could not retrieve active cell count from grid object using get_num_active(). "
                               f"ACTNUM.grdecl indicates {sum_from_actnum_file} active cells. Comparison with grid object not performed.")
        # --- End Active Cell Count Comparison ---

        # --- Test ACTNUM from grid_obj.export_actnum() ---
        if grid_obj and hasattr(grid_obj, 'export_actnum'):
            logger.info(f"[Realization: {realization_dir.name}] Attempting to get ACTNUM via grid_obj.export_actnum().")
            try:
                exported_actnum_obj = grid_obj.export_actnum()
                logger.info(f"[Realization: {realization_dir.name}] grid_obj.export_actnum() returned type: {type(exported_actnum_obj)}")

                exported_actnum_str_to_parse = None
                if isinstance(exported_actnum_obj, str):
                    exported_actnum_str_to_parse = exported_actnum_obj
                elif exported_actnum_obj is not None:
                    logger.info(f"[Realization: {realization_dir.name}] Attempting to convert {type(exported_actnum_obj).__name__} object to string for parsing.")
                    try:
                        converted_str = str(exported_actnum_obj)
                        # Log the type and content of the converted string
                        actual_converted_str_val = str(converted_str) # Ensure it's a string for logging if str() returned non-string
                        log_val_excerpt = actual_converted_str_val[:100].replace(os.linesep, r"\n")
                        logger.info(f"[Realization: {realization_dir.name}] Result of str({type(exported_actnum_obj).__name__}_object): Type={type(converted_str)}, Value (first 100 chars)='{log_val_excerpt}'")
                        
                        if isinstance(converted_str, str) and converted_str.strip(): # Check if it's a non-empty string
                           exported_actnum_str_to_parse = converted_str
                           logger.info(f"[Realization: {realization_dir.name}] Successfully converted object to a non-empty string.")
                        else:
                           logger.warning(f"[Realization: {realization_dir.name}] str({type(exported_actnum_obj).__name__}_object) did not produce a non-empty string (Type: {type(converted_str)}, Is empty/whitespace: {not converted_str.strip() if isinstance(converted_str, str) else 'N/A'}).")
                    except Exception as e_str_conv:
                        logger.error(f"[Realization: {realization_dir.name}] Failed to convert {type(exported_actnum_obj).__name__}_object to string: {e_str_conv}")

                if exported_actnum_str_to_parse and isinstance(exported_actnum_str_to_parse, str):
                    # Prepare the excerpt for logging to avoid complex expressions in f-string
                    debug_log_excerpt = exported_actnum_str_to_parse[:100].replace(os.linesep, r"\n")
                    logger.debug(f"[Realization: {realization_dir.name}] Successfully obtained string for parsing from grid_obj.export_actnum(). First 100 chars: '{debug_log_excerpt}'")
                    
                    # Parse the string data
                    expected_total_cells_for_export = nx * ny * nz
                    actnum_flat_from_export = parse_actnum_string_data(exported_actnum_str_to_parse, expected_total_cells_for_export)
                    
                    if actnum_flat_from_export is not None:
                        if actnum_flat_from_export.size != expected_total_cells_for_export:
                            logger.error(
                                f"[Realization: {realization_dir.name}] ACTNUM from export_actnum(): Parsed data size {actnum_flat_from_export.size} "
                                f"does not match grid dimensions {nx}x{ny}x{nz} ({expected_total_cells_for_export}). Reshape will fail."
                            )
                        else:
                            actnum_3d_from_export = np.transpose(
                                actnum_flat_from_export.reshape((nx, ny, nz), order='F'), (2, 0, 1)
                            )
                            sum_from_export = np.sum(actnum_3d_from_export == 1)
                            logger.info(f"[Realization: {realization_dir.name}] Active cells from grid_obj.export_actnum() (sum of 1s): {sum_from_export}")

                            # Get active count from get_num_active() again for direct comparison here
                            grid_obj_active_count_for_export_comparison = -1
                            if hasattr(grid_obj, 'get_num_active'):
                                try:
                                    grid_obj_active_count_for_export_comparison = grid_obj.get_num_active()
                                except Exception as e_get_active:
                                    logger.warning(f"[Realization: {realization_dir.name}] Could not re-call get_num_active() for export_actnum comparison: {e_get_active}")
                            
                            if grid_obj_active_count_for_export_comparison != -1:
                                logger.info(f"[Realization: {realization_dir.name}] Active cells from grid_obj.get_num_active() (for comparison): {grid_obj_active_count_for_export_comparison}")
                                if sum_from_export == grid_obj_active_count_for_export_comparison:
                                    logger.info(
                                        f"[Realization: {realization_dir.name}] MATCH: Count from export_actnum() ({sum_from_export}) "
                                        f"matches grid_obj.get_num_active() ({grid_obj_active_count_for_export_comparison})."
                                    )
                                else:
                                    logger.error(
                                        f"[Realization: {realization_dir.name}] MISMATCH: Count from export_actnum() ({sum_from_export}) "
                                        f"DOES NOT match grid_obj.get_num_active() ({grid_obj_active_count_for_export_comparison})."
                                    )
                            else:
                                logger.warning(f"[Realization: {realization_dir.name}] Cannot compare export_actnum count as get_num_active() failed for this comparison.")
                    else:
                        logger.warning(f"[Realization: {realization_dir.name}] Failed to parse ACTNUM data from grid_obj.export_actnum() string.")
                else:
                    logger.warning(f"[Realization: {realization_dir.name}] Could not obtain a valid string from grid_obj.export_actnum() for parsing.")
            except Exception as e_export:
                logger.error(f"[Realization: {realization_dir.name}] Error during grid_obj.export_actnum() or its processing: {e_export}")
        else:
            logger.warning(f"[Realization: {realization_dir.name}] Grid object does not have 'export_actnum' method, cannot test this ACTNUM source.")
        # --- End Test ACTNUM from grid_obj.export_actnum() ---

    except ValueError as ve: # Catch ValueErrors from load_actnum or size mismatch
        logger.error(f"ValueError during ACTNUM processing for realization {realization_dir.name}: {ve}. Skipping this realization.")
        raise # Re-raise to be caught by the outer loop's exception handler in prepare_ensemble_training_data
    except Exception as e: # Catch other errors like reshape errors
        logger.error(f"Generic error loading or reshaping ACTNUM from {actnum_file} for realization {realization_dir.name}: {e}. Skipping this realization.")
        raise ValueError(f"Failed to load/process ACTNUM from {actnum_file}") from e
        
    # Load faults and multipliers using the utility function
    fault_array, mult_array = np.zeros((0, 6)), np.zeros((0, 2))
    if fault_file.exists() and mult_file.exists():
        try:
            fault_array, mult_array = load_faults(fault_file, mult_file)
        except Exception as e:
            logger.warning(f"Error loading faults/multipliers: {e}")
    elif fault_file.exists():
        logger.warning(f"Only fault file exists, multiplier file missing. Using empty multiplier data.")
        try:
            with open(fault_file, 'r') as f:
                fault_lines = f.readlines()
                
            # Parse fault definitions
            faults = []
            for line in fault_lines:
                if line.strip() and not line.strip().startswith('--'):
                    parts = line.split()
                    if len(parts) >= 6:  # Expect at least 6 values for fault definition
                        faults.append([float(x) for x in parts[:6]])
                        
            fault_array = np.array(faults)
            mult_array = np.zeros((0, 2))
        except Exception as e:
            logger.warning(f"Error loading faults: {str(e)}. Using empty fault data.")
            fault_array = np.zeros((0, 6))
            mult_array = np.zeros((0, 2))
    elif not 'fault_array' in locals() or not 'mult_array' in locals():
        # If we haven't already defined these variables due to missing files
        fault_array = np.zeros((0, 6))
        mult_array = np.zeros((0, 2))
    
    # --- Use Enhanced EclipseReader --- 
    init_reader = EclipseReader(init_file)
    restart_reader = EclipseReader(restart_file)

    init_data_obj = None
    initial_conditions = None
    dynamic_data = None
    time_array = None
    rate_data = None

    try:
        # --- Load ACTNUM from grdecl file FIRST --- 
        logger.info(f"Loading ACTNUM from {actnum_file}")
        try:
            actnum_flat = load_actnum(actnum_file)
            # Reshape assuming Fortran order, then transpose to (nz, nx, ny)
            actnum = np.transpose(actnum_flat.reshape((nx, ny, nz), order='F'), (2, 0, 1))
            logger.info(f"Loaded and reshaped ACTNUM from grdecl file (shape: {actnum.shape}).")
        except Exception as e:
            logger.error(f"Error loading or reshaping ACTNUM from {actnum_file}: {e}. Cannot proceed without ACTNUM.")
            raise ValueError(f"Failed to load ACTNUM from {actnum_file}") from e
        # --- End Load ACTNUM --- 

        # Load properties from INIT (requires grid dimensions)
        logger.info(f"Loading static properties from {init_file}")
        init_data_obj = init_reader.read_init(grid_obj) # grid_obj has dimensions
        if init_data_obj is None:
             raise ValueError(f"Failed to read INIT file: {init_file}")
        
        # Open restart file and read data
        if not restart_reader.open_restart_file(grid_obj):
             raise ValueError(f"Failed to open UNRST file: {restart_file}")

        initial_conditions = restart_reader.read_initial_conditions()
        if initial_conditions is None:
            raise ValueError(f"Failed to read initial conditions from {restart_file}")

        dynamic_data, time_array = restart_reader.read_dynamic_steps(steppi_indices)
        if dynamic_data is None or time_array is None:
            logger.warning(f"Failed to read some or all dynamic steps from {restart_file}")
            if dynamic_data is None: dynamic_data = {}
            if time_array is None: time_array = np.array([], dtype=np.float32)
            
        # --- Read Production and Injection Rates if requested ---
        if include_rates and data_file is not None:
            logger.info(f"Reading production and injection rates from summary data")
            rate_data = restart_reader.read_summary_rates(data_file, steppi_indices, rate_types)
            logger.info(f"Rate data retrieved: {'Success' if rate_data else 'Failed'}")
            if rate_data:
                logger.info(f"Rate types found: {list(rate_data.keys())}")
        # --- End Read Rates ---

        # --- Generate Fault Transmissibility Multiplier Property ---
        fault_multiplier_array = None
        if fault_file.exists() and mult_file.exists():
            logger.info(f"Creating fault transmissibility multiplier property from {fault_file} and {mult_file}")
            try:
                # Use the new method from EclipseReader
                fault_multiplier_array = init_reader.create_fault_multiplier_array(mult_file, fault_file)
                if fault_multiplier_array is None:
                    logger.warning("Failed to create fault multiplier array. Using all 1.0 values.")
                    fault_multiplier_array = np.ones((nz, nx, ny), dtype=np.float32)
                logger.info(f"Created fault multiplier array with shape {fault_multiplier_array.shape}")
            except Exception as e:
                logger.error(f"Error creating fault multiplier array: {e}")
                logger.warning("Using all 1.0 values for fault multipliers.")
                fault_multiplier_array = np.ones((nz, nx, ny), dtype=np.float32)
        else:
            logger.info("Fault files not found. Using all 1.0 values for fault multipliers.")
            fault_multiplier_array = np.ones((nz, nx, ny), dtype=np.float32)
        # --- End Generate Fault Multiplier Property ---

    finally:
        # Ensure restart file is closed even if errors occurred
        restart_reader.close_restart_file()
    # --- End Enhanced EclipseReader Usage --- 

    # Initialize final data dictionary
    X_data = {}
    
    # --- Check for valid data after reading --- 
    if init_data_obj is None:
        logger.error("INIT data object is None, cannot proceed with data assembly.")
        return X_data # Return empty dict if critical data missing
    if initial_conditions is None:
        logger.error("Initial conditions are None, cannot proceed with data assembly.")
        return X_data
    if dynamic_data is None or time_array is None:
         logger.warning("Dynamic data or time array is None after reading steps.")
         # Ensure they are empty dict/array if None
         dynamic_data = dynamic_data or {}
         time_array = time_array if time_array is not None else np.array([], dtype=np.float32)
    # --- End Check ---
    

    # Add realization ID
    try:
        realization_id = int(realization_dir.name.split('-')[-1])
        X_data['realization_id'] = np.array([realization_id])
    except: X_data['realization_id'] = np.array([0])

    # --- Corrected: Add input variables (static properties from INIT object) ---
    if init_data_obj: # Check if INIT read was successful
        perm_keys = ['PERMX']
        poro_keys = ['PORO']

        # Get Permeability
        perm_found = False
        for key in perm_keys:
            try:
                prop = init_reader.get_property(init_data_obj, key)
                if prop is not None:
                    X_data['permeability'] = prop
                    logger.info(f"Found permeability using key '{key}'")
                    # --- START DEBUG LOGGING FOR PERMEABILITY ---
                    logger.debug(f"[DEBUG] Permeability ({key}) loaded in process_single_realization:")
                    logger.debug(f"  Shape: {prop.shape}")
                    if prop.ndim == 3 and prop.shape[0] > 0 and prop.shape[1] > 0 and prop.shape[2] > 0: # (nz, nx, ny)
                        logger.debug(f"  Overall: Min={np.min(prop)}, Max={np.max(prop)}, Mean={np.mean(prop)}, Zeros={np.count_nonzero(prop == 0)}")
                        layers_to_check = sorted(list(set([0, 1, 14, 15, 16, prop.shape[0]-1])))
                        for i in layers_to_check:
                            if i < prop.shape[0]:
                                layer_data = prop[i, :, :]
                                logger.debug(f"    Layer {i}: Min={np.min(layer_data)}, Max={np.max(layer_data)}, Mean={np.mean(layer_data)}, Zeros={np.count_nonzero(layer_data == 0)}")
                    # --- END DEBUG LOGGING ---
                    perm_found = True
                    break
            except Exception as e:
                logger.warning(f"Could not get static property {key} from INIT: {e}")
        if not perm_found:
             logger.warning(f"Could not find permeability ({perm_keys}) in INIT file {init_file}")

        # Get Porosity
        poro_found = False
        for key in poro_keys:
            try:
                prop = init_reader.get_property(init_data_obj, key)
                if prop is not None:
                    X_data['porosity'] = prop
                    logger.info(f"Found porosity using key '{key}'")
                    # --- START DEBUG LOGGING FOR POROSITY ---
                    logger.debug(f"[DEBUG] Porosity ({key}) loaded in process_single_realization:")
                    logger.debug(f"  Shape: {prop.shape}")
                    if prop.ndim == 3 and prop.shape[0] > 0 and prop.shape[1] > 0 and prop.shape[2] > 0: # (nz, nx, ny)
                        logger.debug(f"  Overall: Min={np.min(prop)}, Max={np.max(prop)}, Mean={np.mean(prop)}, Zeros={np.count_nonzero(prop == 0)}")
                        layers_to_check = sorted(list(set([0, 1, 14, 15, 16, prop.shape[0]-1])))
                        for i in layers_to_check:
                            if i < prop.shape[0]:
                                layer_data = prop[i, :, :]
                                logger.debug(f"    Layer {i}: Min={np.min(layer_data)}, Max={np.max(layer_data)}, Mean={np.mean(layer_data)}, Zeros={np.count_nonzero(layer_data == 0)}")
                    # --- END DEBUG LOGGING ---
                    poro_found = True
                    break
            except Exception as e:
                logger.warning(f"Could not get static property {key} from INIT: {e}")
        if not poro_found:
            logger.warning(f"Could not find porosity ({poro_keys}) in INIT file {init_file}")
    else:
        logger.error("Cannot extract static properties because INIT data object is None.")
    # --- End Correction ---

    # Add fault data - Replace the old fault_array with the new fault_multiplier_array
    X_data['Fault'] = fault_multiplier_array  # 3D array with shape (nz, nx, ny) matching other static properties

    # Add initial conditions from UNRST
    if initial_conditions:
        if 'PRESSURE' in initial_conditions:
            X_data['Pini'] = initial_conditions['PRESSURE']
        if 'SWAT' in initial_conditions:
            X_data['Sini'] = initial_conditions['SWAT']
        # Add others like SGAS, SOIL if read

    # Add dynamic data from UNRST
    if dynamic_data:
        if 'PRESSURE' in dynamic_data:
            X_data['Pressure'] = dynamic_data['PRESSURE']
        if 'SWAT' in dynamic_data:
            X_data['Water_saturation'] = dynamic_data['SWAT']
        if 'SGAS' in dynamic_data:
            X_data['Gas_saturation'] = dynamic_data['SGAS']
        if 'SOIL' in dynamic_data:
            X_data['Oil_saturation'] = dynamic_data['SOIL']
            
    # --- Add Production and Injection Rates Data ---
    if rate_data:
        # Map rate data types to more descriptive names in X_data
        rate_mapping = {
            'COPR': 'Qo',                  # Oil production rate
            'CWPR': 'Qw',                  # Water production rate 
            'CGPR': 'Qg',                  # Gas production rate
            'CWIR': 'Water_injection_rate',
            'CGIR': 'Gas_injection_rate',
            'CVPR': 'Qvp',                 # Volume production rate
            'CVIR': 'Qvi',                 # Volume injection rate
            'Q': 'Q'                       # Total production rate
        }
        
        for rate_type, mapped_name in rate_mapping.items():
            if rate_type in rate_data:
                X_data[mapped_name] = rate_data[rate_type]
                logger.info(f"Added {mapped_name} data with shape {rate_data[rate_type].shape}")
    else:
        # Create zero-filled rate arrays to ensure consistent shape across realizations
        logger.warning("No rate data found, creating zero-filled placeholders for consistency")
        
        # Get the expected shape for rate arrays from time_array
        num_steps = len(time_array) if time_array is not None else steppi
        
        rate_mapping = {
            'COPR': 'Qo',                  # Oil production rate
            'CWPR': 'Qw',                  # Water production rate 
            'CGPR': 'Qg',                  # Gas production rate
            'CWIR': 'Water_injection_rate',
            'CGIR': 'Gas_injection_rate',
            'CVPR': 'Qvp',                 # Volume production rate
            'CVIR': 'Qvi',                 # Volume injection rate
            'Q': 'Q'                       # Total production rate
        }
        
        # Create zero arrays for each rate type
        if rate_types is None:
            # Default to all rate types
            curr_rate_types = ["COPR", "CWPR", "CGPR", "CWIR", "CGIR", "CVPR", "CVIR", "Q"]
        else:
            curr_rate_types = rate_types + ["Q"]  # Always include total rate
            
        for rate_type in curr_rate_types:
            if rate_type in rate_mapping:
                mapped_name = rate_mapping[rate_type]
                X_data[mapped_name] = np.zeros((num_steps, nz, nx, ny), dtype=np.float32)
                logger.info(f"Added zero placeholder for {mapped_name} with shape {X_data[mapped_name].shape}")
    # --- End Add Rates ---

    # Add Time array from UNRST
    if time_array is not None:
        X_data['Time'] = time_array
    else:
        X_data['Time'] = np.array([], dtype=np.float32)

    # Add other necessary variables
    X_data['actnum'] = actnum
    # Add report steps that were actually read
    num_valid_steps = len(X_data.get('Time', [])) # Use length of time array as indicator
    actual_report_steps = [step for i, step in enumerate(steppi_indices) if i < num_valid_steps] # Infer from time array length
    X_data['report_steps'] = np.array(actual_report_steps, dtype=np.int32)

    # --- Validation --- 
    logger.info(f"Validating shapes for realization: {realization_dir.name}")
    num_valid_steps = len(X_data.get('Time', []))

    # Check static properties (3D)
    for prop in ['permeability', 'porosity', 'Pini', 'Sini', 'actnum']:
        if prop in X_data:
            prop_shape = X_data[prop].shape
            if len(prop_shape) != 3 or prop_shape != (nx, ny, nz):
                 logger.warning(f"Shape mismatch for static prop {prop}: expected {(nx, ny, nz)}, got {prop_shape}.")
                 # Attempt reshape...

    # Check dynamic properties (4D: steps, nx, ny, nz)
    dynamic_props = ['Pressure', 'Water_saturation', 'Gas_saturation', 'Oil_saturation',
                     'Qo', 'Qw', 'Qg', 'Q',
                     'Water_injection_rate', 'Gas_injection_rate', 
                     'Qvp', 'Qvi']
                     
    for prop in dynamic_props:
        if prop in X_data:
            prop_shape = X_data[prop].shape
            expected_shape = (num_valid_steps, nz, nx, ny)
            if len(prop_shape) != 4 or prop_shape != expected_shape:
                 logger.warning(f"Shape mismatch for dynamic prop {prop}: expected {expected_shape}, got {prop_shape}.")
                 # Attempt reshape...

    # Check Fault shape - Now a 3D property like other static properties
    if 'Fault' in X_data:
        fault_shape = X_data['Fault'].shape
        expected_fault_shape = (nz, nx, ny)
        if len(fault_shape) != 3 or fault_shape != expected_fault_shape:
            logger.warning(f"Shape mismatch for Fault: expected {expected_fault_shape}, got {fault_shape}")
    
    # Check Time and report_steps length
    if 'Time' in X_data and len(X_data['Time']) != num_valid_steps:
         logger.warning(f"Time array length {len(X_data['Time'])} doesn't match valid steps read {num_valid_steps}")
    if 'report_steps' in X_data and len(X_data['report_steps']) != num_valid_steps:
        logger.warning(f"report_steps array length {len(X_data['report_steps'])} doesn't match valid steps read {num_valid_steps}")

    # Final check for required props
    required_props = ['permeability', 'porosity']
    missing_props = [prop for prop in required_props if prop not in X_data]
    if missing_props:
        logger.warning(f"Missing required properties for realization {realization_dir.name}: {missing_props}")
        
    logger.info(f"Successfully processed data for realization: {realization_dir.name}")

    # --- Create and add effective actnum based on grid_obj's active cells ---
    if grid_obj and hasattr(grid_obj, 'getNumActive') and all([nx, ny, nz]):
        try:
            num_effective_active = grid_obj.getNumActive()
            logger.info(f"Number of effective active cells from grid_obj.getNumActive(): {num_effective_active}")
            if num_effective_active > 0:
                active_cell_markers = np.ones(num_effective_active, dtype=np.int8) # Use int8 for a 0/1 mask
                # map_active_to_full_grid is imported from eclipse_reader
                actnum_effective_mapped = map_active_to_full_grid(
                    active_data=active_cell_markers, 
                    grid_obj=grid_obj, 
                    nx=nx, 
                    ny=ny, 
                    nz=nz, 
                    default_val=0  # Inactive cells will be 0
                )
                if actnum_effective_mapped is not None:
                    X_data['actnum_effective'] = actnum_effective_mapped
                    logger.info(f"Successfully created 'actnum_effective' with shape {actnum_effective_mapped.shape}. Non-zero elements: {np.count_nonzero(actnum_effective_mapped)}")
                else:
                    logger.warning("Failed to create 'actnum_effective' because mapping returned None.")
            else:
                logger.warning("grid_obj.getNumActive() returned 0, creating empty 'actnum_effective'.")
                X_data['actnum_effective'] = np.zeros((nz, nx, ny), dtype=np.int8)
        except Exception as e_eff_actnum:
            logger.error(f"Error creating 'actnum_effective': {e_eff_actnum}")
            # Fallback to empty if error occurs
            X_data['actnum_effective'] = np.zeros((nz, nx, ny), dtype=np.int8)
    else:
        logger.warning("Cannot create 'actnum_effective': grid_obj not available or dimensions not set.")
        X_data['actnum_effective'] = np.zeros((nz, nx, ny), dtype=np.int8) # Fallback
    # --- End effective actnum ---

    return X_data

def extract_well_connections(dframe):
    """Extract well names and their connection grid blocks from the DataFrame.
    
    Args:
        dframe: DataFrame containing connection data from summary file
        
    Returns:
        Dictionary mapping well names to lists of their connection grid blocks (i,j,k)
    """
    well_connections = {}
    connection_count = 0
    
    # Extract well names and connection indexes from column names
    for column in dframe.columns:
        parts = column.split(':')
        if len(parts) < 3:
            continue
            
        # Format: "COPR:WELL_NAME:i,j,k" or sometimes with an integer instead of i,j,k
        rate_type = parts[0]
        well_name = parts[1]
        cell_indices = parts[2]
        
        if ',' in cell_indices:
            try:
                i, j, k = map(int, cell_indices.split(','))
                # Adjust for 1-based indexing in Eclipse
                i -= 1
                j -= 1
                k -= 1
                
                if well_name not in well_connections:
                    well_connections[well_name] = []
                    
                # Add unique connection point
                connection = (i, j, k)
                if connection not in well_connections[well_name]:
                    well_connections[well_name].append(connection)
                    connection_count += 1
            except ValueError:
                # Skip if can't parse indices
                pass
    
    # Log well connection information
    logger.info(f"Extracted {connection_count} connection points for {len(well_connections)} wells")
    
    # Log the first few wells and their connections (for debugging)
    if well_connections:
        well_list = list(well_connections.keys())
        sample_size = min(3, len(well_list))
        
        for i in range(sample_size):
            well = well_list[i]
            conn_count = len(well_connections[well])
            logger.info(f"Well '{well}' has {conn_count} connection points")
            
            # Log a few connection points for each well
            if conn_count > 0:
                sample_conn = min(5, conn_count)
                conn_str = ", ".join([f"({i},{j},{k})" for i, j, k in well_connections[well][:sample_conn]])
                if conn_count > sample_conn:
                    conn_str += f", ... and {conn_count - sample_conn} more"
                logger.info(f"  Sample connections for '{well}': {conn_str}")
    
    return well_connections

def build_peaceman_input_data(
    all_data,
    well_connections_by_realization,
    num_realizations,
    steppi,
    nx, ny, nz,
    well_types=None
):
    """Build input data array for Peaceman well model.
    
    Args:
        all_data: Dictionary containing all simulation data
        well_connections_by_realization: Dictionary mapping realization index to well connection info
        num_realizations: Number of realizations
        steppi: Number of timesteps
        nx, ny, nz: Grid dimensions
        well_types: Dictionary mapping well names to their types (producer, water_injector, gas_injector)
        
    Returns:
        Dictionary containing X_data2 with X input array
    """
    logger.info("Building Peaceman well model input data (X_data2)")
    
    # Create empty X_data2 dictionary
    X_data2 = {}
    
    # First, identify all unique well names across all realizations
    all_well_names = set()
    for real_idx, well_conn in well_connections_by_realization.items():
        all_well_names.update(well_conn.keys())
    
    all_well_names = sorted(list(all_well_names))
    num_all_wells = len(all_well_names)
    logger.info(f"Found {num_all_wells} unique wells across all realizations")
    
    # Extract producer wells from well_types
    producer_wells = []
    special_keys = ['__original_names__', '__normalized_types__']
    
    if well_types:
        # Get producers using well_types
        producer_wells = [w for w, t in well_types.items() 
                       if t == 'producer' and w not in special_keys]
        
        # Get normalized well names to types mapping if available
        normalized_types = well_types.get('__normalized_types__', {})
        
        # Also check for wells that might be producers by normalized name
        for well_name in all_well_names:
            if well_name not in producer_wells:
                # Try matching with normalized name
                normalized_name = normalize_well_name(well_name)
                well_type = normalized_types.get(normalized_name)
                if well_type == 'producer':
                    producer_wells.append(well_name)
    
    # Filter well_names to only include producers that exist in connections
    producer_wells = [w for w in producer_wells if w in all_well_names]
    
    # Sort producers alphabetically
    producer_wells = sorted(producer_wells)
    N_pr = len(producer_wells)
    
    if N_pr == 0:
        logger.warning("No producer wells found! Using all wells instead.")
        producer_wells = all_well_names
        N_pr = len(producer_wells)
    
    logger.info(f"Using {N_pr} producer wells out of {num_all_wells} total wells")
    
    # Log well names
    if N_pr > 0:
        logger.info(f"Producer wells: {', '.join(producer_wells)}")
        # Also log in groups of 10 if there are many wells
        if N_pr > 20:
            for i in range(0, N_pr, 10):
                end_idx = min(i + 10, N_pr)
                well_group = producer_wells[i:end_idx]
                logger.info(f"Producers {i+1}-{end_idx}: {', '.join(well_group)}")
    else:
        logger.warning("No producer wells found in any realization")
    
    # Store well names in the dictionary
    X_data2['well_names'] = producer_wells
    
    # Store well types if available
    if well_types:
        # Create a mask array for each well type
        producer_mask = np.ones(N_pr, dtype=np.int32)  # All wells are producers now
        
        # Store masks in the dictionary
        X_data2['producer_mask'] = producer_mask
        X_data2['producer_wells'] = producer_wells
    
    # Calculate feature size: 4 properties per well + global pressure + timestep
    feature_size = N_pr * 4 + 1 + 1
    logger.info(f"Feature size: {feature_size} (4 properties × {N_pr} wells + pressure + timestep)")
    
    # Create empty X array
    X = np.zeros((num_realizations, steppi, feature_size), dtype=np.float32)
    
    # Required properties with alternative key mappings
    required_keys = {
        'permeability': ['permeability', 'PERMX', 'PERM'],
        'Water_saturation': ['Water_saturation', 'SWAT', 'Sw'],
        'Oil_saturation': ['Oil_saturation', 'SOIL', 'So'],
        'Gas_saturation': ['Gas_saturation', 'SGAS', 'Sg'],
        'Pressure': ['Pressure', 'PRESSURE', 'P']
    }
    
    # Check for missing properties and try alternative keys
    property_map = {}
    missing_keys = []
    for main_key, alt_keys in required_keys.items():
        found = False
        for key in alt_keys:
            if key in all_data:
                property_map[main_key] = key
                found = True
                break
        if not found:
            missing_keys.append(main_key)
    
    # Special handling for oil saturation - can be calculated as 1 - Sw - Sg
    if 'Oil_saturation' in missing_keys and 'Water_saturation' in property_map and 'Gas_saturation' in property_map:
        logger.info("Oil saturation not directly available - will calculate from water and gas saturations")
        # We'll calculate it on-the-fly, remove from missing keys
        missing_keys.remove('Oil_saturation')
        # Mark for calculation with None
        property_map['Oil_saturation'] = None
    
    # Check if any critical properties are still missing
    critical_missing = [k for k in missing_keys if k != 'Gas_saturation']  # Gas might be optional
    if critical_missing:
        logger.error(f"Missing critical properties for X_data2: {critical_missing}")
        raise ValueError(f"Cannot build X_data2 without properties: {critical_missing}")
    elif missing_keys:
        logger.warning(f"Missing non-critical properties for X_data2: {missing_keys}. Will use zeros.")
    
    logger.info(f"Using property mapping: {property_map}")
    
    # Process each realization
    for real_idx in range(num_realizations):
        if real_idx not in well_connections_by_realization:
            logger.warning(f"No well connection data for realization {real_idx}. Using zeros.")
            continue
            
        # Get well connections for this realization
        well_connections = well_connections_by_realization[real_idx]
        
        # For each timestep
        for time_idx in range(steppi):
            feature_idx = 0
            
            # Process each producer well
            for well_name in producer_wells:
                # If well exists in this realization
                if well_name in well_connections and well_connections[well_name]:
                    connections = well_connections[well_name]
                    
                    # Extract permeability (static property)
                    perm_values = []
                    perm_key = property_map.get('permeability')
                    if perm_key:
                        for i, j, k in connections:
                            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                                # Permeability is stored as (num_realizations, 1, nz, nx, ny)
                                if all_data[perm_key][real_idx, 0, k, i, j] > 0:
                                    perm_values.append(all_data[perm_key][real_idx, 0, k, i, j])
                    
                    # Calculate average permeability along wellbore
                    avg_perm = np.mean(perm_values) if perm_values else 0.0
                    X[real_idx, time_idx, feature_idx] = avg_perm
                    feature_idx += 1
                    
                    # Extract dynamic properties for the current timestep (saturations)
                    sat_keys = ['Oil_saturation', 'Water_saturation', 'Gas_saturation']
                    
                    for sat_prop in sat_keys:
                        sat_key = property_map.get(sat_prop)
                        
                        if sat_key is not None:
                            # Direct property available
                            prop_values = []
                            for i, j, k in connections:
                                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                                    # Dynamic properties stored as (num_realizations, time, nz, nx, ny)
                                    if time_idx < all_data[sat_key].shape[1]:  # Check time dimension
                                        prop_values.append(all_data[sat_key][real_idx, time_idx, k, i, j])
                            
                            # Calculate average property along wellbore
                            avg_prop = np.mean(prop_values) if prop_values else 0.0
                            X[real_idx, time_idx, feature_idx] = avg_prop
                        
                        elif sat_prop == 'Oil_saturation' and property_map.get('Oil_saturation') is None:
                            # Calculate oil saturation as 1 - Sw - Sg
                            water_key = property_map.get('Water_saturation')
                            gas_key = property_map.get('Gas_saturation')
                            
                            water_values = []
                            gas_values = []
                            
                            for i, j, k in connections:
                                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                                    if time_idx < all_data[water_key].shape[1]:
                                        water_values.append(all_data[water_key][real_idx, time_idx, k, i, j])
                                    
                                    if gas_key and time_idx < all_data[gas_key].shape[1]:
                                        gas_values.append(all_data[gas_key][real_idx, time_idx, k, i, j])
                            
                            avg_water = np.mean(water_values) if water_values else 0.0
                            avg_gas = np.mean(gas_values) if gas_values else 0.0
                            
                            # Calculate oil saturation, ensuring it's within [0,1]
                            avg_oil = max(0.0, min(1.0, 1.0 - avg_water - avg_gas))
                            X[real_idx, time_idx, feature_idx] = avg_oil
                        
                        else:
                            # Property not available, use zero
                            X[real_idx, time_idx, feature_idx] = 0.0
                        
                        feature_idx += 1
                else:
                    # Skip 4 features for this missing well
                    feature_idx += 4
            
            # Add global average pressure for this timestep
            press_key = property_map.get('Pressure')
            if press_key and time_idx < all_data[press_key].shape[1]:
                # Calculate global average pressure excluding inactive/zero cells
                pressure_data = all_data[press_key][real_idx, time_idx]
                global_avg_pressure = np.mean(pressure_data[pressure_data > 0]) if np.any(pressure_data > 0) else 0.0
                X[real_idx, time_idx, feature_idx] = global_avg_pressure
                feature_idx += 1
            else:
                X[real_idx, time_idx, feature_idx] = 0.0
                feature_idx += 1
            
            # Add timestep (normalized by total steps)
            X[real_idx, time_idx, feature_idx] = time_idx / max(1, steppi - 1)
    
    # Store X array in dictionary
    X_data2['X'] = X
    X_data2['Y'] = np.zeros((num_realizations, steppi, N_pr), dtype=np.float32)  # Placeholder empty Y
    
    # Add metadata
    X_data2['grid_dimensions'] = np.array([nx, ny, nz])
    X_data2['num_wells'] = N_pr
    X_data2['num_timesteps'] = steppi
    X_data2['feature_size'] = feature_size
    
    logger.info(f"Built X_data2 with X shape: {X.shape}")
    return X_data2

def scale_dict_arrays(
    data_dict: Dict[str, np.ndarray],
    config: Optional[Dict[str, float]] = None,
    is_peaceman_data: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Scale arrays in dictionary to appropriate ranges for PINO model.
    
    Args:
        data_dict: Dictionary containing numpy arrays
        config: Optional dictionary with scaling parameters (min/max values for different fields)
        is_peaceman_data: Whether this is X_data2 (Peaceman model data)
    
    Returns:
        Tuple of (dictionary with scaled arrays, dictionary with scaling parameters used)
    """
    # Initialize scaling parameters dictionary
    scaling_params = {}
    
    # Initialize config if None
    if config is None:
        config = {}
    
    # Create a copy of the dictionary to avoid modifying the original
    scaled_dict = {}
    for key, value in data_dict.items():
        # For non-array or non-numeric types, copy directly
        if not isinstance(value, np.ndarray) or value.dtype.kind not in 'fc':
            scaled_dict[key] = value
            continue
        else:
            scaled_dict[key] = value.copy()
    
    # Define scalar scaling parameters (for X_data1)
    if not is_peaceman_data:
        scaling_fields = {
            'permeability': ('minK', 'maxK'),
            'Pini': ('minP', 'maxP'),
            'Pressure': ('minP', 'maxP'),  # Use same scaling as Pini
            'Qw': ('minQW', 'maxQW'),
            'Qg': ('minQg', 'maxQg'),
            'Q': ('minQ', 'maxQ')
        }
        
        # Calculate min/max values for each field that needs scaling
        for field, (min_key, max_key) in scaling_fields.items():
            if field in scaled_dict:
                # Get min/max from config or calculate from data
                if min_key in config:
                    min_val = config[min_key]
                    logger.info(f"Using configured {min_key}={min_val} for {field}")
                else:
                    # Get only finite, non-NaN, non-zero values for min calculation
                    valid_data = scaled_dict[field][np.isfinite(scaled_dict[field]) & (scaled_dict[field] != 0)]
                    min_val = float(np.min(valid_data)) if valid_data.size > 0 else 0.0
                    logger.info(f"Calculated {min_key}={min_val} for {field}")
                
                if max_key in config:
                    max_val = config[max_key]
                    logger.info(f"Using configured {max_key}={max_val} for {field}")
                else:
                    # Get only finite, non-NaN values for max calculation
                    valid_data = scaled_dict[field][np.isfinite(scaled_dict[field])]
                    max_val = float(np.max(valid_data)) if valid_data.size > 0 else 1.0
                    logger.info(f"Calculated {max_key}={max_val} for {field}")
                
                # Store in scaling parameters
                scaling_params[field] = {
                    'min': min_val,
                    'max': max_val
                }
                
                # Apply scaling
                if min_val != max_val:
                    scaled_dict[field] = (scaled_dict[field] - min_val) / (max_val - min_val)
                    # Clip to ensure [0, 1] range (handling potential numeric issues)
                    scaled_dict[field] = np.clip(scaled_dict[field], 0.0, 1.0)
                    logger.info(f"Scaled {field} to range [0, 1]")
                else:
                    logger.warning(f"Cannot scale {field} - min and max values are identical: {min_val}")
                    scaled_dict[field][:] = 0.0
    else:
        # Peaceman data (X_data2) processing
        # X has shape (num_realizations, steppi, feature_size)
        if 'X' in scaled_dict:
            x_data = scaled_dict['X']
            num_features = x_data.shape[2]
            scaling_params['X'] = []
            
            # Scale each feature column separately
            for i in range(num_features):
                feature_data = x_data[:, :, i]
                
                # Get column name if available in metadata
                column_name = f"feature_{i}"
                
                # Check if we have a named parameter for this feature
                config_key_min = None
                config_key_max = None
                
                # Determine if this is a specific known feature we want to map to config keys
                if 'num_wells' in scaled_dict and i < scaled_dict['num_wells']:
                    # This might be a permeability feature (1st feature per well)
                    if i % 4 == 0:
                        config_key_min = 'minK'
                        config_key_max = 'maxK'
                        column_name = f"perm_well_{i//4}"
                    # This might be the pressure feature (last feature)
                    elif i == num_features - 2:
                        config_key_min = 'minP'
                        config_key_max = 'maxP'
                        column_name = "global_pressure"
                
                # Get min/max values (from config if available, otherwise from data)
                if config_key_min in config:
                    min_val = config[config_key_min]
                    logger.info(f"Using configured {config_key_min}={min_val} for X column {i} ({column_name})")
                else:
                    valid_values = feature_data[np.isfinite(feature_data) & (feature_data != 0)]
                    min_val = float(np.min(valid_values)) if valid_values.size > 0 else 0.0
                    logger.info(f"Calculated min={min_val} for X column {i} ({column_name})")
                
                if config_key_max in config:
                    max_val = config[config_key_max]
                    logger.info(f"Using configured {config_key_max}={max_val} for X column {i} ({column_name})")
                else:
                    valid_values = feature_data[np.isfinite(feature_data)]
                    max_val = float(np.max(valid_values)) if np.any(np.isfinite(valid_values)) else 1.0
                    logger.info(f"Calculated max={max_val} for X column {i} ({column_name})")
                
                # Store parameters for this feature
                scaling_params['X'].append({
                    'min': min_val,
                    'max': max_val,
                    'name': column_name
                })
                
                # Apply scaling
                if min_val != max_val:
                    x_data[:, :, i] = (feature_data - min_val) / (max_val - min_val)
                    x_data[:, :, i] = np.clip(x_data[:, :, i], 0.0, 1.0)
                else:
                    logger.warning(f"Cannot scale X column {i} - min and max values are identical: {min_val}")
                    x_data[:, :, i] = 0.0
            
            # Update the data dictionary with scaled values
            scaled_dict['X'] = x_data
        
        # Similarly for Y if present (production rates)
        if 'Y' in scaled_dict:
            y_data = scaled_dict['Y']
            scaling_params['Y'] = []
            
            # Y is typically shape (num_realizations, steppi, 3*num_producers)
            # Where each producer has 3 values: oil, water, gas rates
            if len(y_data.shape) > 2:
                num_outputs = y_data.shape[2]
                
                for i in range(num_outputs):
                    output_data = y_data[:, :, i]
                    
                    # Get output name if available
                    output_name = f"output_{i}"
                    if 'output_metrics' in scaled_dict and i // 3 < len(scaled_dict['output_metrics']):
                        metric_idx = i % 3
                        well_idx = i // 3
                        if metric_idx == 0:
                            output_name = f"oil_rate_well_{well_idx}"
                            config_key_min = None  # Could map to minQ if desired
                            config_key_max = None  # Could map to maxQ if desired
                        elif metric_idx == 1:
                            output_name = f"water_rate_well_{well_idx}"
                            config_key_min = 'minQW'
                            config_key_max = 'maxQW'
                        elif metric_idx == 2:
                            output_name = f"gas_rate_well_{well_idx}"
                            config_key_min = 'minQg'
                            config_key_max = 'maxQg'
                    
                    # Get min/max values (from config if available)
                    if config_key_min in config:
                        min_val = config[config_key_min]
                        logger.info(f"Using configured {config_key_min}={min_val} for Y column {i} ({output_name})")
                    else:
                        valid_values = output_data[np.isfinite(output_data) & (output_data != 0)]
                        min_val = float(np.min(valid_values)) if valid_values.size > 0 else 0.0
                        logger.info(f"Calculated min={min_val} for Y column {i} ({output_name})")
                    
                    if config_key_max in config:
                        max_val = config[config_key_max]
                        logger.info(f"Using configured {config_key_max}={max_val} for Y column {i} ({output_name})")
                    else:
                        valid_values = output_data[np.isfinite(output_data)]
                        max_val = float(np.max(valid_values)) if np.any(np.isfinite(valid_values)) else 1.0
                        logger.info(f"Calculated max={max_val} for Y column {i} ({output_name})")
                    
                    # Store parameters
                    scaling_params['Y'].append({
                        'min': min_val,
                        'max': max_val,
                        'name': output_name
                    })
                    
                    # Apply scaling
                    if min_val != max_val:
                        y_data[:, :, i] = (output_data - min_val) / (max_val - min_val)
                        y_data[:, :, i] = np.clip(y_data[:, :, i], 0.0, 1.0)
                    else:
                        logger.warning(f"Cannot scale Y column {i} - min and max values are identical: {min_val}")
                        y_data[:, :, i] = 0.0
            
            # Update the data dictionary with scaled values
            scaled_dict['Y'] = y_data
    
    # Final pass: verify all arrays have the correct data type
    for key in scaled_dict:
        if key == 'Time':
            # Ensure Time is float32 but skip scaling
            if isinstance(scaled_dict[key], np.ndarray) and scaled_dict[key].dtype.kind in 'fc':
                scaled_dict[key] = scaled_dict[key].astype(np.float32)
            continue
        
        if isinstance(scaled_dict[key], np.ndarray) and scaled_dict[key].dtype.kind in 'fc':
            # Only convert numeric arrays (float/complex), not integers like grid dimensions
            scaled_dict[key] = scaled_dict[key].astype(np.float32)
    
    return scaled_dict, scaling_params

def prepare_ensemble_training_data(
    ensemble_dir: Union[str, Path],
    output_file: Union[str, Path],
    nx: int,
    ny: int,
    nz: int,
    steppi: int,
    steppi_indices: List[int],
    realization_pattern: str = "realization-*",
    max_realizations: Optional[int] = None,
    include_rates: bool = True,
    rate_types: Optional[List[str]] = None,
    peaceman_output_file: Optional[Union[str, Path]] = None,
    scaling_config_file: Optional[Union[str, Path]] = None
) -> None:
    """Prepare training data from an ensemble of Eclipse simulations.
    
    Args:
        ensemble_dir: Directory containing realization folders
        output_file: Path to save output pickle file
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        nz: Number of cells in z direction
        steppi: Number of timesteps
        steppi_indices: List of report steps to include from UNRST file
        realization_pattern: Pattern to match realization directories
        max_realizations: Maximum number of realizations to process (None for all)
        include_rates: Whether to include production/injection rates
        rate_types: List of rate types to include (None for all)
        peaceman_output_file: Path to save Peaceman model input data (optional)
        scaling_config_file: Path to YAML file with scaling parameters (optional)
    """
    try:
        # Load scaling configuration if provided
        if scaling_config_file:
            from scaling_utils import load_scaling_config
            scaling_config = load_scaling_config(scaling_config_file)
        else:
            scaling_config = {}
        
        ensemble_dir = Path(ensemble_dir)
        logger.info(f"Processing ensemble in directory: {ensemble_dir}")
        
        # Verify that steppi matches the length of steppi_indices
        if len(steppi_indices) != steppi:
            logger.warning(f"steppi ({steppi}) does not match the length of steppi_indices ({len(steppi_indices)}). Using the length of steppi_indices as steppi.")
            steppi = len(steppi_indices)
        
        logger.info(f"Using steppi={steppi} with {len(steppi_indices)} report steps: {steppi_indices}")
        
        # Find all realization directories
        realization_dirs = list(ensemble_dir.glob(realization_pattern))
        if not realization_dirs:
            raise ValueError(f"No realization directories found matching pattern '{realization_pattern}' in {ensemble_dir}")
        
        # Sort realizations numerically by extracting the number from realization-N format
        def get_realization_number(path):
            try:
                # Extract the number after the last dash
                return int(path.name.split('-')[-1])
            except (ValueError, IndexError):
                # If can't parse number, return a large number to sort to the end
                return float('inf')
        
        # Sort realizations numerically
        realization_dirs = sorted(realization_dirs, key=get_realization_number)
        
        logger.info(f"Found {len(realization_dirs)} realization directories")
        
        if max_realizations is not None and max_realizations > 0:
            realization_dirs = realization_dirs[:max_realizations]
            logger.info(f"Limited to {max_realizations} realizations: {[dir.name for dir in realization_dirs]}")
        
        # Find SCHEDULE file in first realization to extract well types
        well_types = {}
        try:
            if realization_dirs:
                first_realization = realization_dirs[0]
                
                # Create a temporary reader to help find the SCHEDULE file
                temp_reader = EclipseReader(first_realization / "dummy.txt")
                schedule_file = temp_reader.find_schedule_file(first_realization)
                
                if schedule_file:
                    logger.info(f"Parsing well types from SCHEDULE file: {schedule_file}")
                    well_types = temp_reader.read_well_types_from_schedule(schedule_file)
                    logger.info(f"Parsed {len(well_types)} well types from SCHEDULE file")
                else:
                    logger.warning("No SCHEDULE file found")
        except Exception as e:
            logger.error(f"Error parsing well types from SCHEDULE file: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Log well types summary 
        if well_types:
            # Skip the special metadata entries when counting
            special_keys = ['__original_names__', '__normalized_types__']
            
            # Get producers using only original well names (not normalized duplicates)
            producers = [w for w, t in well_types.items() 
                        if t == 'producer' and w not in special_keys]
            water_injectors = [w for w, t in well_types.items() 
                              if t == 'water_injector' and w not in special_keys]
            gas_injectors = [w for w, t in well_types.items() 
                            if t == 'gas_injector' and w not in special_keys]
            
            # Get original names mapping if available
            original_names = well_types.get('__original_names__', {})
            
            logger.info(f"Well types summary: {len(producers)} unique producers, "
                       f"{len(water_injectors)} water injectors, {len(gas_injectors)} gas injectors")
            
            if producers:
                logger.info(f"Producer wells: {', '.join(producers[:min(20, len(producers))])}")
                if len(producers) > 20:
                    logger.info(f"... and {len(producers) - 20} more producers")
            else:
                logger.warning("No producer wells found in SCHEDULE file!")
        else:
            logger.warning("No well types parsed from SCHEDULE file!")
        
        # Dictionary to hold all realization data
        all_data = {}
        
        # Dictionary to hold well connection data for X_data2 (Peaceman)
        well_connections_by_realization = {}
        
        # Dictionary to hold well production data for Y output
        well_production_by_realization = {}
        
        # Process each realization
        for i, realization_dir in enumerate(realization_dirs):
            try:
                logger.info(f"Processing realization {i+1}/{len(realization_dirs)}: {realization_dir.name}")
                
                # For X_data2 (Peaceman), we need to collect well connection data
                # First, find DATA file
                data_file = None
                if include_rates or peaceman_output_file:
                    data_files = list(realization_dir.glob("*.DATA"))
                    if not data_files:
                        # Try case-insensitive search
                        data_files = list(realization_dir.glob("*.data"))
                        
                    if data_files:
                        data_file = data_files[0]
                        logger.info(f"Found DATA file for rates/connections: {data_file}")
                
                # Process the realization normally
                realization_data = process_single_realization(
                    realization_dir=realization_dir,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    steppi=steppi,
                    steppi_indices=steppi_indices,
                    include_rates=include_rates,
                    rate_types=rate_types
                )
                
                # Extract well connections and production data for Peaceman model if needed
                if peaceman_output_file and data_file:
                    try:
                        # Create a reader to get well connections
                        grid_file = list(realization_dir.glob("*.EGRID"))[0]
                        temp_reader = EclipseReader(grid_file)
                        grid_obj = temp_reader.read_grid()
                        
                        # Import required modules
                        try:
                            import res2df
                            from res2df import summary, ResdataFiles
                            
                            # Create ResdataFiles object
                            resdatafiles = ResdataFiles(str(data_file))
                            
                            # Get connection data (C* vectors) as a pandas DataFrame
                            connection_df = summary.df(resdatafiles, column_keys="C*", time_index="raw")
                            
                            if not connection_df.empty:
                                # Log connection data details
                                connection_cols = list(connection_df.columns)
                                logger.info(f"Connection data has {len(connection_cols)} columns and {len(connection_df)} rows")
                                logger.debug(f"First few connection columns: {connection_cols[:min(10, len(connection_cols))]}")
                                
                                # Extract well connections
                                well_connections = extract_well_connections(connection_df)
                                well_connections_by_realization[i] = well_connections
                                logger.info(f"Extracted connections for {len(well_connections)} wells in realization {i}")
                                
                                # Log the well names found in connections
                                conn_well_names = sorted(list(well_connections.keys()))
                                logger.info(f"Wells with connections in realization {i}: {', '.join(conn_well_names[:min(20, len(conn_well_names))])}")
                                
                                # Extract production data using the EclipseReader
                                try:
                                    # Get available steps from connection data for this realization
                                    available_steps = list(range(len(connection_df)))
                                    
                                    # Use the extract_well_production_data method from EclipseReader
                                    production_data = temp_reader.extract_well_production_data(
                                        data_file,
                                        available_steps,
                                        steppi_indices,
                                        well_types,
                                        producer_only=True
                                    )
                                    
                                    well_production_by_realization[i] = production_data
                                    
                                    logger.info(f"Extracted production data for {len(production_data)} wells in realization {i}")
                                    if production_data:
                                        logger.info(f"Wells with production data: {', '.join(sorted(list(production_data.keys())))}")
                                except Exception as prod_err:
                                    logger.error(f"Error extracting production data for realization {i}: {prod_err}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                            else:
                                logger.warning(f"No connection data found for realization {i}")
                        except ImportError:
                            logger.warning("res2df module not available, cannot extract well connections and production data")
                        except Exception as conn_err:
                            logger.warning(f"Error extracting connections for realization {i}: {conn_err}")
                            import traceback
                            logger.error(traceback.format_exc())
                    except Exception as e:
                        logger.warning(f"Could not process well connections for realization {i}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Add realization data to overall dataset
                for key, value in realization_data.items():
                    if key not in all_data:
                        # Initialize with list for first realization
                        all_data[key] = [value]
                    else:
                        # Append for subsequent realizations
                        all_data[key].append(value)
                
                logger.info(f"Successfully processed realization: {realization_dir.name}")
                
            except Exception as e:
                logger.error(f"Error processing realization {realization_dir.name}: {str(e)}")
                logger.error("Skipping this realization and continuing with others")
        
        # Log summary information about well connections
        if peaceman_output_file and well_connections_by_realization:
            all_wells = set()
            wells_by_realization = {}
            
            for real_idx, wells in well_connections_by_realization.items():
                wells_by_realization[real_idx] = set(wells.keys())
                all_wells.update(wells.keys())
            
            logger.info(f"Found a total of {len(all_wells)} unique wells across all realizations")
            
            # Log production data summary
            if well_production_by_realization:
                all_prod_wells = set()
                for real_idx, wells_data in well_production_by_realization.items():
                    all_prod_wells.update(wells_data.keys())
                
                logger.info(f"Found production data for {len(all_prod_wells)} wells across all realizations")
                logger.info(f"Wells with production data: {', '.join(sorted(list(all_prod_wells)))}")
            else:
                logger.warning("No production data found in any realization!")
        
        # Process all_data to create X_data1 (regular dataset)
        logger.info("Processing all collected data to create main dataset (X_data1)")
        num_realizations = len(realization_dirs)
        
        # Create standard X_data1 dictionary
        X_data1 = {}
        
        # Process realization IDs if available
        if 'realization_id' in all_data:
            realization_ids = np.concatenate(all_data['realization_id'])
            X_data1['realization_id'] = realization_ids
            logger.info(f"Added realization_id data with shape {realization_ids.shape}")
        
        # Process static and dynamic data
        static_keys = ['permeability', 'porosity', 'Pini', 'Sini', 'actnum', 'Fault', 'actnum_effective']
        dynamic_keys = ['Pressure', 'Water_saturation', 'Gas_saturation', 'Oil_saturation',
                      'Qo', 'Qw', 'Qg', 'Q', 'Water_injection_rate', 'Gas_injection_rate',
                      'Qvp', 'Qvi']
        
        # Process and stack all data
        for key in all_data:
            if key == 'realization_id':
                continue  # Already processed
                
            if key in ['report_steps']:
                # Use the first realization's value for these
                X_data1[key] = all_data[key][0]
                logger.info(f"Using {key} data from first realization with shape {X_data1[key].shape}")
            elif key == 'Time':
                # Special handling for Time array - get the time values
                time_values = all_data[key][0]  # 1D array of timesteps
                logger.info(f"Original Time array shape: {time_values.shape}")
                
                # Create a 5D array (N, steppi, nz, nx, ny) with timestep values repeated
                num_realizations = len(all_data[key])
                time_5d = np.zeros((num_realizations, steppi, nz, nx, ny), dtype=np.float32)
                
                # Fill the array with timestep values (same for each realization and gridblock)
                for t in range(steppi):
                    if t < len(time_values):
                        # Use the actual time value if available
                        time_value = time_values[t]
                    else:
                        # Fallback - use timestep index if not enough values
                        time_value = float(t)
                    
                    # Set the value for all grid blocks for this timestep
                    time_5d[:, t, :, :, :] = time_value
                
                # Store in X_data1
                X_data1['Time'] = time_5d
                logger.info(f"Created Time array with shape {time_5d.shape}, filled with timestep values")
            else:
                # Stack arrays for all realizations
                try:
                    stacked = np.stack(all_data[key], axis=0)
                    logger.info(f"Stacked {key} data with shape {stacked.shape}")
                    
                    # --- START DEBUG LOGGING FOR STACKED PERM/PORO ---
                    if key in ['permeability', 'porosity']:
                        logger.debug(f"[DEBUG] Stacked {key} in prepare_ensemble_training_data (before potential expand_dims):")
                        logger.debug(f"  Shape: {stacked.shape}")
                        # Assuming stacked shape is (num_real, nz, nx, ny) for these properties
                        if stacked.ndim == 4 and stacked.shape[0] > 0 and stacked.shape[1] > 0:
                            first_real_data = stacked[0, :, :, :] # Data for the first realization
                            logger.debug(f"  First Realization Overall: Min={np.min(first_real_data)}, Max={np.max(first_real_data)}, Mean={np.mean(first_real_data)}, Zeros={np.count_nonzero(first_real_data == 0)}")
                            layers_to_check = sorted(list(set([0, 1, 14, 15, 16, first_real_data.shape[0]-1]))) # first_real_data.shape[0] is nz
                            for i in layers_to_check:
                                if i < first_real_data.shape[0]:
                                    layer_data = first_real_data[i, :, :]
                                    logger.debug(f"    Layer {i} (Realization 0): Min={np.min(layer_data)}, Max={np.max(layer_data)}, Mean={np.mean(layer_data)}, Zeros={np.count_nonzero(layer_data == 0)}")
                    # --- END DEBUG LOGGING ---

                    # Add time dimension to static properties if needed
                    if key in static_keys and stacked.ndim == 4:  # (num_real, nz, nx, ny)
                        X_data1[key] = np.expand_dims(stacked, axis=1)  # Add time dim
                        # --- START DEBUG LOGGING FOR EXPANDED PERM/PORO ---
                        if key in ['permeability', 'porosity']:
                            expanded_prop = X_data1[key]
                            logger.debug(f"[DEBUG] Expanded {key} in prepare_ensemble_training_data (after expand_dims):")
                            logger.debug(f"  Shape: {expanded_prop.shape}")
                            # Assuming expanded shape is (num_real, 1, nz, nx, ny)
                            if expanded_prop.ndim == 5 and expanded_prop.shape[0] > 0 and expanded_prop.shape[2] > 0:
                                first_real_data_expanded = expanded_prop[0, 0, :, :, :] # Data for the first realization, first time step
                                logger.debug(f"  First Realization (Expanded) Overall: Min={np.min(first_real_data_expanded)}, Max={np.max(first_real_data_expanded)}, Mean={np.mean(first_real_data_expanded)}, Zeros={np.count_nonzero(first_real_data_expanded == 0)}")
                                layers_to_check = sorted(list(set([0, 1, 14, 15, 16, first_real_data_expanded.shape[0]-1]))) # first_real_data_expanded.shape[0] is nz
                                for i in layers_to_check:
                                    if i < first_real_data_expanded.shape[0]:
                                        layer_data = first_real_data_expanded[i, :, :]
                                        logger.debug(f"    Layer {i} (Realization 0, Expanded): Min={np.min(layer_data)}, Max={np.max(layer_data)}, Mean={np.mean(layer_data)}, Zeros={np.count_nonzero(layer_data == 0)}")
                        # --- END DEBUG LOGGING ---
                        logger.info(f"Added time dimension to {key}, new shape {X_data1[key].shape}")
                    else:
                        X_data1[key] = stacked
                except Exception as e:
                    logger.error(f"Error stacking {key} data: {e}")
        
        # Clean data and ensure valid values
        logger.info("Cleaning data in X_data1")
        X_data1 = clean_dict_arrays(X_data1)
        
        # Scale data arrays for PINO model
        logger.info("Scaling data in X_data1 for PINO model")
        X_data1, scaling_params1 = scale_dict_arrays(X_data1, config=scaling_config)
        
        # Save X_data1 to file
        logger.info(f"Saving main dataset to {output_file}")
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(X_data1, f)
        
        # Save scaling parameters to JSON for reference
        import json
        scaling_params_file = Path(output_file).with_suffix('.scaling.json')
        with open(scaling_params_file, 'w') as f:
            json.dump(scaling_params1, f, indent=2)
        logger.info(f"Saved scaling parameters to {scaling_params_file}")
        
        # Create and save X_data2 for Peaceman model if requested
        if peaceman_output_file:
            try:
                logger.info(f"Building Peaceman well model input data for {peaceman_output_file}")
                
                # Build X_data2
                X_data2 = build_peaceman_input_data(
                    X_data1,
                    well_connections_by_realization,
                    num_realizations,
                    steppi,
                    nx, ny, nz,
                    well_types=well_types
                )
                
                # Populate Y data with production data for producer wells if available
                if well_production_by_realization:
                    # Extract all producer wells from well_types
                    producer_wells = [well for well, wtype in well_types.items() 
                                       if wtype == 'producer' and well != '__original_names__']
                    
                    # If producer_wells list is available in X_data2, use that
                    if 'producer_wells' in X_data2:
                        producer_wells = X_data2['producer_wells']
                    
                    # Get all well names from X_data2
                    all_well_names = X_data2.get('well_names', [])
                    
                    if all_well_names and producer_wells:
                        # Find indices of producer wells in the all_well_names list
                        producer_indices = [all_well_names.index(well) for well in producer_wells if well in all_well_names]
                        
                        # Create a Y array with the right producers if we have production data
                        if well_production_by_realization:
                            # Define the production metrics to extract
                            output_metrics = ['oil_rate', 'water_rate', 'gas_rate', 'oil_total', 'gas_oil_ratio', 'water_cut']
                            num_metrics = len(output_metrics)
                            
                            # Make sure num_wells is defined (number of unique wells)
                            num_wells = len(all_well_names)
                            
                            # Create a temporary Y array with all metrics for all wells
                            Y_temp = np.zeros((num_realizations, steppi, num_wells, num_metrics), dtype=np.float32)
                            
                            # Fill in the Y array with production data
                            for real_idx, well_data in well_production_by_realization.items():
                                if real_idx >= num_realizations:
                                    continue  # Skip if realization index is out of bounds
                                    
                                for well_name, metrics in well_data.items():
                                    if well_name in all_well_names:
                                        well_idx = all_well_names.index(well_name)
                                        
                                        # Fill in each metric for this well
                                        for metric_idx, metric_name in enumerate(output_metrics):
                                            if metric_name in metrics:
                                                for time_idx in range(min(steppi, len(metrics[metric_name]))):
                                                    Y_temp[real_idx, time_idx, well_idx, metric_idx] = metrics[metric_name][time_idx]
                            
                            # Reshape Y to focus only on producer wells and their oil, water, and gas rates
                            # Find indices of producer wells
                            producer_indices = [all_well_names.index(well) for well in producer_wells if well in all_well_names]
                            num_producers = len(producer_indices)
                            
                            # Create a new Y array with shape (N, steppi, 3*num_producers)
                            # Only selecting oil_rate, water_rate, gas_rate (first 3 metrics)
                            Y_reshaped = np.zeros((num_realizations, steppi, 3 * num_producers), dtype=np.float32)
                            
                            # Metrics to include (only the production rates)
                            rate_metrics = ['oil_rate', 'water_rate', 'gas_rate']
                            rate_indices = [output_metrics.index(metric) for metric in rate_metrics]
                            
                            # Fill the reshaped Y array
                            for prod_idx, well_idx in enumerate(producer_indices):
                                for metric_idx, rate_idx in enumerate(rate_indices):
                                    # Map each production rate to its position in the flattened array
                                    flat_idx = prod_idx * 3 + metric_idx
                                    # Copy the data from the original Y array
                                    Y_reshaped[:, :, flat_idx] = Y_temp[:, :, well_idx, rate_idx]
                            
                            # Store the Y array in X_data2
                            X_data2['Y'] = Y_reshaped
                            X_data2['output_metrics'] = rate_metrics
                            X_data2['producer_indices'] = producer_indices
                            
                            # Calculate some statistics for logging
                            non_zero_values = np.count_nonzero(Y_reshaped)
                            total_elements = Y_reshaped.size
                            non_zero_percentage = 100 * non_zero_values / max(1, total_elements)
                            
                            logger.info(f"Added production data to Y array with shape {Y_reshaped.shape}")
                            logger.info(f"Y array contains {non_zero_values} non-zero values out of {total_elements} elements ({non_zero_percentage:.2f}%)")
                            
                            # Log some sample values for verification
                            if non_zero_values > 0:
                                # Find a non-zero value to show as example
                                for real_idx in range(min(3, num_realizations)):
                                    for producer_idx, well_idx in enumerate(producer_indices):
                                        well_name = all_well_names[well_idx]
                                        for metric_idx, metric_name in enumerate(rate_metrics):
                                            flat_idx = producer_idx * 3 + metric_idx
                                            # Get max value for this well and metric
                                            max_val = np.max(Y_reshaped[real_idx, :, flat_idx])
                                            if max_val > 0:
                                                logger.info(f"Sample: Realization {real_idx}, Well {well_name}, {metric_name} max value: {max_val}")
                                                break
                        else:
                            # No production data available, create empty Y array with new shape format
                            # Find indices of producer wells
                            producer_indices = [all_well_names.index(well) for well in producer_wells if well in all_well_names]
                            num_producers = len(producer_indices)
                            
                            # Create a placeholder Y with shape (N, steppi, 3*num_producers)
                            Y_empty = np.zeros((num_realizations, steppi, 3 * num_producers), dtype=np.float32)
                            X_data2['Y'] = Y_empty
                            X_data2['output_metrics'] = ['oil_rate', 'water_rate', 'gas_rate']
                            X_data2['producer_indices'] = producer_indices
                            
                            logger.warning("No production data available for Y. Created empty array with shape: " + 
                                          f"{Y_empty.shape} for {num_producers} producer wells")
                
                # Clean data arrays for Peaceman model
                logger.info("Cleaning data in X_data2")
                X_data2 = clean_dict_arrays(X_data2)
                
                # Scale data arrays for Peaceman model
                logger.info("Scaling data in X_data2 for PINO model")
                X_data2, scaling_params2 = scale_dict_arrays(X_data2, config=scaling_config, is_peaceman_data=True)
                
                # Save X_data2 to file
                logger.info(f"Saving Peaceman well model data to {peaceman_output_file}")
                with gzip.open(peaceman_output_file, 'wb') as f:
                    pickle.dump(X_data2, f)
                
                # Save Peaceman scaling parameters
                peaceman_scaling_file = Path(peaceman_output_file).with_suffix('.scaling.json')
                with open(peaceman_scaling_file, 'w') as f:
                    json.dump(scaling_params2, f, indent=2)
                logger.info(f"Saved Peaceman scaling parameters to {peaceman_scaling_file}")
                
                logger.info(f"Successfully created Peaceman well model data")
            except Exception as e:
                logger.error(f"Error creating Peaceman well model data: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"Error preparing ensemble training data: {str(e)}")
        raise

def parse_actnum_string_data(actnum_string_data: str, expected_total_cells: Optional[int] = None) -> Optional[np.ndarray]:
    """Parse ACTNUM data from a multi-line string (e.g., from grid_obj.export_actnum()).

    Args:
        actnum_string_data: The multi-line string containing ACTNUM data.
        expected_total_cells: Optional. If provided, the parsed count from the header
                              will be checked against this number.

    Returns:
        A flat numpy array of ACTNUM values, or None if parsing fails critically.
    """
    logger.debug("Attempting to parse ACTNUM data from string.")
    lines = actnum_string_data.strip().splitlines()
    values = []
    actnum_data_started = False
    parsed_count_from_header = -1

    if not lines:
        logger.error("ACTNUM string data is empty.")
        return None

    # First line is expected to be the header, e.g., "ACTNUM    3320344 INTE"
    header_line = lines[0].strip().upper()
    if header_line.startswith("ACTNUM"):
        parts = header_line.split()
        if len(parts) >= 2:
            try:
                parsed_count_from_header = int(parts[1])
                logger.debug(f"Parsed count from ACTNUM string header: {parsed_count_from_header}")
                if expected_total_cells is not None and parsed_count_from_header != expected_total_cells:
                    logger.warning(
                        f"Count in ACTNUM string header ({parsed_count_from_header}) "
                        f"does not match expected total cells ({expected_total_cells})."
                    )
            except ValueError:
                logger.warning(f"Could not parse count from ACTNUM string header: '{header_line}'. Proceeding without header count check.")
        else:
            logger.warning(f"ACTNUM string header format unexpected: '{header_line}'. Proceeding without header count check.")
        # Data starts from the next line
        data_lines = lines[1:]
    else:
        logger.warning("First line of ACTNUM string data does not look like a standard ACTNUM header. Assuming all lines are data.")
        data_lines = lines # Process all lines as data if no clear header

    for line_num, line_content in enumerate(data_lines):
        line = line_content.strip()
        if not line or line.startswith('--'):  # Skip empty lines and comments
            continue
        if line == '/':  # End of section
            logger.debug(f"Found end of ACTNUM data in string at relative line {line_num + 1}.")
            break

        for item_idx, item in enumerate(line.split()):
            if not item: continue
            # Skip items that are just a series of dots (e.g., '....')
            if item.strip() == '' or all(c == '.' for c in item.strip()):
                logger.debug(f"Skipping non-data item '{item}' in string data at line {line_num + 1}, item {item_idx + 1}.")
                continue

            if '*' in item:
                parts = item.split('*')
                if len(parts) == 2:
                    try:
                        count = int(parts[0])
                        value = int(parts[1])
                        if count < 0:
                            logger.warning(f"Negative count '{count}' in multiplier '{item}' in string data at line {line_num + 1}, item {item_idx + 1}. Interpreting as 0.")
                            count = 0
                        values.extend([value] * count)
                    except ValueError:
                        logger.error(f"Invalid number in multiplier '{item}' in string data at line {line_num + 1}, item {item_idx + 1}. Skipping item.")
                else:
                    logger.warning(f"Malformed multiplier '{item}' in string data at line {line_num + 1}, item {item_idx + 1}. Skipping item.")
            else:
                try:
                    values.append(int(item))
                except ValueError:
                    logger.error(f"Invalid integer value '{item}' in string data at line {line_num + 1}, item {item_idx + 1}. Skipping item.")
    
    if not values:
        logger.warning("No ACTNUM values parsed from the provided string data.")
        return np.array([], dtype=int) # Return empty if no values, might be valid if 0 cells

    logger.info(f"Successfully parsed {len(values)} values from ACTNUM string data.")
    
    # Optional: Final check against header count if available and reliable
    if parsed_count_from_header != -1 and len(values) != parsed_count_from_header:
        logger.warning(f"Number of parsed values ({len(values)}) from string data does not match count in its header ({parsed_count_from_header}).")
        # Depending on strictness, could return None here

    return np.array(values)

if __name__ == "__main__":
    # Base directory containing realizations
    ensemble_dir = "/home/azureuser/cloudfiles/code/Users/admin.bjorn.egil.ludvigsen/JOHAN_SVERDRUP/FMU"
    
    # Output file path
    output_file = "/home/azureuser/cloudfiles/code/Users/admin.bjorn.egil.ludvigsen/JOHAN_SVERDRUP/FMU/js_data_train.pkl.gz"
    
    # Output file path for Peaceman well model
    peaceman_output_file = "/home/azureuser/cloudfiles/code/Users/admin.bjorn.egil.ludvigsen/JOHAN_SVERDRUP/FMU/js_data_train_peaceman.pkl.gz"    
    # Grid dimensions and timesteps
    nx, ny, nz = 106, 164, 191  # grid dimensions
    
    # Define report steps to use
    steppi = 12  # number of report steps to train on
    #steppi_indices = [6, 32, 54, 82, 95, 114, 140, 182, 241, 302, 372, 444]  # Specific report steps to use. not all existing.
    steppi_indices = [10, 32, 55, 82, 97, 114, 154, 182, 228, 302, 395, 444]  # Specific report steps to use
    # Available restart reports  on JS FMU: [0, 10, 32, 33, 55, 56, 71, 75, 82, 92, 97, 98, 102, 
    #  114, 124, 154, 182, 228, 263, 301, 302, 351, 395, 444, 
    # 452, 454, 462, 469, 478, 484, 492, 498, 508, 515, 522, 528, 539, 544, 549, 554, 559, 563]

    # Physical constraints before scaling
    pre_scaling_thresholds = {
        'porosity': (0.0, 1.0),
        'Pressure': (0.0, 500.0),  # Example physical pressure range (bar)
        'Water_saturation': (0.0, 1.0),
        'Gas_saturation': (0.0, 1.0),
        'Oil_saturation': (0.0, 1.0),
        'permeability': (0.0, 5000.0)  # Example physical permeability range (mD)
    }
    
    # Normalized constraints after scaling
    post_scaling_thresholds = {
        # All values should be in [0,1] range after scaling
        'porosity': (0.0, 1.0),
        'Pressure': (0.0, 1.0),
        'Water_saturation': (0.0, 1.0),
        'Gas_saturation': (0.0, 1.0),
        'Oil_saturation': (0.0, 1.0),
        'permeability': (0.0, 1.0)
    }
    
    # Fields to ensure non-negative values
    fields_to_clip = [
        'permeability', 'porosity', 'Fault', 'Pini', 'Sini',
        'Pressure', 'Water_saturation', 'Gas_saturation', 'Oil_saturation',
        'Q', 'Qo', 'Qw', 'Qg',
        'Water_injection_rate', 'Gas_injection_rate', 'Qvp', 'Qvi'
    ]
    
    # Define which rate types to include
    rate_types = ["COPR", "CWPR", "CGPR", "CWIR", "CGIR", "CVPR", "CVIR"]
    
    # Process ensemble
    prepare_ensemble_training_data(
        ensemble_dir=ensemble_dir,
        output_file=output_file,
        nx=nx,
        ny=ny,
        nz=nz,
        steppi=steppi,
        steppi_indices=steppi_indices,
        realization_pattern="realization-*",
        max_realizations=8,  # Set a limit if needed, None for all realizations
        include_rates=True,  # Include production and injection rates
        rate_types=rate_types,  # Specify which rate types to include
        peaceman_output_file=peaceman_output_file,  # Add output file for Peaceman well model
        scaling_config_file=None  # No scaling config file provided
    ) 

    logger.info("Script finished.") 
