"""Experimental constants and condition mappings; SOA codes require log validation."""
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Primary subject manifest: columns subject,path,group.
METADATA_FILE = PROJECT_DIR / "meta_data.csv"
INPUT_DIR = None
OUTPUT_DIR = PROJECT_DIR / "RSA_results"

# Optional trial-count validation CSV; None to disable.
COUNTS_FILE = None

# EEG data load parameters
SEMANTIC = ('PA', 'PC', 'A', 'C', 'AC', 'U')
SOA_CODES = (1, 2)  # Must verify whether 1=150 ms and 2=300 ms in the experiment log.
EVENT_TO_PAIR = {
    **{33+i:(SEMANTIC[i],1) for i in range(6)},
    **{49+i:(SEMANTIC[i],2) for i in range(6)},
}
PSEUDOWORD_CODES = {39,55}
STIM_CODES = set(EVENT_TO_PAIR) | PSEUDOWORD_CODES
EOG_NAMES = {'HEOG1','HEOG2','VEOG1','VEOG2'}
MASTOIDS = {'M1','M2'}
GROUPS = {'GJ':0,'GA':1}
PAIRS = np.triu_indices(6,1)

# RSA parameters
BLOCK_SIZE = 104
WINDOW_SAMPLES = 1
STEP_SAMPLES = 1
SHRINK = 0.15
RANK_TOL = 1e-7
MIN_PER_CELL = 1
INCLUDE_MASTOIDS = False
PARTITION_MODE = "odd_even"  # or "strict_odd_even"
N_PERM = 2000
LIMIT = None  # set to 1 for initial testing; None = all participants
SEED = 2026
# H1/H2 are Related-only theoretical distances; H3 is A-specific.
# H4 uses U only to define AC-A-C+U interaction vector.

# Optional H2/H4 sensitivity analysis (run sensitivity_h2_h4.py after main.py)
SENS_INPUT_DIR = OUTPUT_DIR  # Directory containing subject *_rsa.npz outputs
SENS_OUTPUT_DIR = PROJECT_DIR / 'RSA_H2_H4_sensitivity'
SENS_SMOOTH_SAMPLES = (3, 5)  # centered boxcar on computed H2/H4 values
SENS_N_PERM = 2000
SENS_N_BOOT = 2000
SENS_SOA_INDEX = 0  # SOA code 1
SENS_SHIFT_ANALYSIS_MS = (150.0, 500.0)
SENS_MAX_LAG_MS = 150.0
SENS_MIN_OVERLAP = 20
SENS_SEED = 2026
