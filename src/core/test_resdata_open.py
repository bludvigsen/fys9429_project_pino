# test_resdata_open.py
import resdata.grid
import resdata.resfile
from pathlib import Path
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- !! IMPORTANT: Set the correct paths below !! ---
realization_dir = Path("S:/GRIEG_AASEN/Petrel/User/bjolud/PHD/JOHAN_SVERDRUP/FMU/realization-0")
egrid_filename = "25P0P3_HISTANDPRED_FF_16022025-0.EGRID" # <-- Make sure this is the correct EGRID filename
unrst_filename = "25P0P3_HISTANDPRED_FF_16022025-0.UNRST"
# --- End Path Configuration ---

grid_path = realization_dir / egrid_filename
unrst_path = realization_dir / unrst_filename

grid_obj = None
rst_file = None

try:
    logging.info(f"Attempting to read grid: {grid_path}")
    if not grid_path.exists():
        logging.error(f"EGRID file not found at: {grid_path}")
        exit()
    grid_obj = resdata.grid.Grid(str(grid_path)) # Grid constructor usually takes str()
    logging.info(f"Grid read successfully: {grid_obj.nx}x{grid_obj.ny}x{grid_obj.nz}")

    logging.info(f"Attempting to open restart file: {unrst_path}")
    if not unrst_path.exists():
            logging.error(f"UNRST file not found at: {unrst_path}")
            exit()

    # Try opening with str(path) first
    try:
        logging.info(f"Trying with str(): {str(unrst_path)}")
        rst_file = resdata.resfile.ResdataRestartFile(grid_obj, str(unrst_path))
        logging.info("Restart file opened successfully with str().")
    except Exception as e_str:
        logging.warning(f"Opening with str() failed: {e_str}")
        # If str() failed, try with as_posix()
        try:
                logging.info(f"Trying with as_posix(): {unrst_path.as_posix()}")
                rst_file = resdata.resfile.ResdataRestartFile(grid_obj, unrst_path.as_posix())
                logging.info("Restart file opened successfully with as_posix().")
        except Exception as e_posix:
            logging.error(f"Opening with as_posix() also failed: {e_posix}")
            raise e_posix # Raise the error from the second attempt if both fail

    logging.info(f"Restart file content accessible. Available steps: {rst_file.report_steps}")
    logging.info(f"Restart file content accessible. Available dates: {rst_file.report_dates}")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    traceback.print_exc()
finally:
    if rst_file:
        try:
            rst_file.close()
            logging.info("Restart file closed.")
        except Exception as close_e:
            logging.error(f"Error closing restart file: {close_e}")

#try:
#    # Ensure the file exists right before opening
#    if not unrst_path.exists():
#        logger.error(f"File does not exist at path: {self.unrst_path}")
#        return False           
#    # Use ResdataRestartFile with grid_obj first and the original mapped drive path string
#    rst_file = resfile.ResdataRestartFile(grid_obj, unrst_path)
#    report_dates = rst_file.report_dates
#    available_steps = rst_file.report_steps
#    logging.info(f"Successfully opened restart file. Available steps: {available_steps}")
#    return True
#except Exception as e:
#    logging.error(f"Error opening restart file {unrst_path}: {str(e)}")
#    rst_file = None
#    return False
#
#if rst_file:
#    try:
#        rst_file.close()
#        logging.info("Restart file closed.")
#    except Exception as close_e:
#        logging.error(f"Error closing restart file: {close_e}")

logging.info("Test script finished.")