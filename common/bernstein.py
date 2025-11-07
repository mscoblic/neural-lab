import numpy as np
from pathlib import Path

# Import BeBOT exactly like your script does
# We assume the original BeBOT is still in tools/extra inside the repo
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'tools' / 'extra'))
from common.BeBOT import PiecewiseBernsteinPoly


def build_control_points(start, end, heading_cp, preds):
    """
    Build 10 control points (x,y,z individually) for the Bernstein curve.
    Matches your current logic exactly: cp5 is duplicated.
    """

    # preds shape = (7, 3): cp2..cp8 OR (6,3) if heading_cp is cp2
    preds = np.asarray(preds)

    if preds.shape[0] == 7:
        # Case: model predicts cp2..cp8 directly
        cp2, cp3, cp4, cp5, cp6, cp7, cp8 = preds
    else:
        # Case: you use heading_cp as cp2, and predictions give cp3..cp8
        cp2 = heading_cp
        cp3, cp4, cp5, cp6, cp7, cp8 = preds

    # Build arrays exactly like your plotting code
    cx = np.array([start[0], cp2[0], cp3[0], cp4[0], cp5[0],
                   cp5[0], cp6[0], cp7[0], cp8[0], end[0]])

    cy = np.array([start[1], cp2[1], cp3[1], cp4[1], cp5[1],
                   cp5[1], cp6[1], cp7[1], cp8[1], end[1]])

    cz = np.array([start[2], cp2[2], cp3[2], cp4[2], cp5[2],
                   cp5[2], cp6[2], cp7[2], cp8[2], end[2]])

    return cx, cy, cz


def eval_piecewise_curve(cx, cy, cz, n_eval=50, tknots=(0.0, 0.5, 1.0)):
    """
    Evaluate the piecewise Bernstein polynomial using BeBOT.
    Matches behavior in your script 1-for-1.
    """

    tknots = np.array(tknots, dtype=float)
    t_eval = np.linspace(0, 1, n_eval)

    tx = PiecewiseBernsteinPoly(cx, tknots, t_eval)[0, :]
    ty = PiecewiseBernsteinPoly(cy, tknots, t_eval)[0, :]
    tz = PiecewiseBernsteinPoly(cz, tknots, t_eval)[0, :]

    return tx, ty, tz
