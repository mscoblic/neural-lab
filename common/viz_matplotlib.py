import numpy as np
import matplotlib.pyplot as plt
from common.utils import to_np
from common.bernstein import build_control_points, eval_piecewise_curve


def plot_dataset_sample(
    X_denorm,
    Y_true_denorm,
    Y_pred_denorm,
    idx,
    *,
    save_path=None,
    title_prefix="sample",
    n_eval=50,
    elev=45,
    azim=-70,
    obstacle_radius=0.1
):
    """
    Static 3D matplotlib plot for a single trajectory sample.
    X_denorm: (4,3) tokens: start/end/obs/heading
    Y_true_denorm: (T_out,3)
    Y_pred_denorm: (T_out,3)
    """

    # unpack tokens
    start = X_denorm[0]
    end   = X_denorm[1]
    obs   = X_denorm[2]
    head  = X_denorm[3]

    # extract predicted CPs (model predictions)
    preds = Y_pred_denorm

    # build CPs with cp5 duplication
    cx, cy, cz = build_control_points(start, end, head, preds)

    # evaluate trajectory
    tx, ty, tz = eval_piecewise_curve(cx, cy, cz, n_eval=n_eval)

    # begin figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # continuous curve
    ax.plot(tx, ty, tz, 'b-', linewidth=2, alpha=0.7, label='trajectory')

    # tokens
    ax.scatter(*start, s=80, label='start')
    ax.scatter(*end,   s=120, marker='*', label='end')
    ax.scatter(*obs,   s=120, color='red', label='obstacle center')

    # ground truth
    if Y_true_denorm is not None:
        Yt = to_np(Y_true_denorm)
        ax.scatter(Yt[:,0], Yt[:,1], Yt[:,2], s=30,
                   color='black', marker='x', label='ground truth')

    # predicted CPs
    Yp = to_np(Y_pred_denorm)
    ax.scatter(Yp[:,0], Yp[:,1], Yp[:,2], s=28,
               marker='o', label='prediction')

    # label predicted CPs
    for i,(x,y,z) in enumerate(Yp):
        ax.text(float(x), float(y), float(z), str(i+1),
                fontsize=8, color='blue')

    # obstacle sphere
    R = obstacle_radius
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = obs[0] + R*np.cos(u)*np.sin(v)
    ys = obs[1] + R*np.sin(u)*np.sin(v)
    zs = obs[2] + R*np.cos(v)
    ax.plot_surface(xs, ys, zs, color='red', alpha=0.3, linewidth=0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{title_prefix} idx={idx}")
    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
