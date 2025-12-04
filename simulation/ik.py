import numpy as np

def ik(limb, target_pos, target_ori_y, q0, T_chest_world):
    """
    limb: one of ["right_arm", "left_arm", "right_leg", "left_leg"]
    """

    if limb == "right_arm":
        return ik_right_arm(target_pos, target_ori_y, q0, T_chest_world)

    elif limb == "left_arm":
        return ik_left_arm(target_pos, target_ori_y, q0, T_chest_world)

    elif limb == "right_leg":
        return ik_right_leg(target_pos, q0, T_chest_world)

#    elif limb == "left_leg":
#       return ik_left_leg(target_pos, q0, T_chest_world)

    else:
        raise ValueError(f"Unknown limb: {limb}")


def ik_right_arm(target_pos, target_ori_y, q0, T_chest_world):

    max_iter = 60
    tol = 1e-4

    # --- Joint Limits (example) ---
    q_min = np.deg2rad(np.array([-90, -60,   0, -130]))
    q_max = np.deg2rad(np.array([ 90, 120, 132,   35]))

    # --- Reichweitenprüfung ---
    # Schulterposition aus Brust-FK extrahieren
    shoulder_world = (T_chest_world @ np.array([-0.0586, 0.021, 0.0330, 1]))[:3]
    if np.linalg.norm(target_pos - shoulder_world) > 0.21:
        return q0, False   # klar unerreichbar

    q = q0.copy()
    alpha = 3.0
    prev_err = 1e9

    for _ in range(max_iter):

        # ---- FK ----
        q_full = np.zeros(16)
        q_full[:4] = q
        T = fk(q_full, T_chest_world)["hand_r"]
        pos = T[:3, 3]

        e_pos = target_pos - pos
        err = np.linalg.norm(e_pos)

        # 1) Erfolg
        if err < tol:
            return q, True

        # 2) Keine Verbesserung -> nicht erreichbar
        if abs(prev_err - err) < 1e-6:
            return q, False

        prev_err = err

        # ---- Position-Jacobian ----
        eps = 1e-6
        J_pos = np.zeros((3,4))
        for i in range(4):
            dq = q.copy()
            dq[i] += eps
            q_full_eps = np.zeros(16)
            q_full_eps[:4] = dq
            T_eps = fk(q_full_eps, T_chest_world)["hand_r"]
            pos_eps = T_eps[:3, 3]
            J_pos[:, i] = (pos_eps - pos) / eps

        # ---- DLS Pseudoinverse ----
        lam = 1e-4
        J_pinv = np.linalg.inv(J_pos.T @ J_pos + lam*np.eye(4)) @ J_pos.T

        # ---- Primärbewegung ----
        dq_pos = J_pinv @ e_pos

        # ---- Nullraum-Orientierung ----
        e_ori = (q[1] + q[2] + q[3]) - target_ori_y
        grad = np.array([0.0, 1.0, 1.0, 1.0]) * (-e_ori)
        N = np.eye(4) - J_pinv @ J_pos
        dq_ori = N @ grad * alpha

        # ---- Gesamtupdate ----
        q += dq_pos + dq_ori

        # ---- Gelenklimits ----
        q = np.clip(q, q_min, q_max)

        # 3) Gelenk an Limit & Fehler groß => nicht erreichbar
        touching_limits = np.any((q <= q_min+1e-4) | (q >= q_max-1e-4))
        if touching_limits and err > 0.005:
            return q, False

    # Kein Erfolg nach max_iter
    return q, False


def ik_left_arm(target_pos, target_ori_y, q0, T_chest_world):

    max_iter = 60
    tol = 1e-4

    # --- Joint Limits (example, mirror of right arm) ---
    q_min = np.deg2rad(np.array([-90, -60,   0, -130]))
    q_max = np.deg2rad(np.array([ 90, 120, 132,   35]))

    # --- Reichweitenprüfung (linke Schulter) ---
    shoulder_offset_l = np.array([0.0586, 0.021, 0.0330, 1.0])
    shoulder_world = (T_chest_world @ shoulder_offset_l)[:3]

    if np.linalg.norm(target_pos - shoulder_world) > 0.21:
        return q0, False

    q = q0.copy()
    alpha = 3.0
    prev_err = 1e9

    for _ in range(max_iter):

        # ---- FK ----
        q_full = np.zeros(16)
        # linker Arm beginnt bei q_full[4:8]
        q_full[4:8] = q

        T = fk(q_full, T_chest_world)["hand_l"]
        pos = T[:3, 3]

        e_pos = target_pos - pos
        err = np.linalg.norm(e_pos)

        # Erfolg
        if err < tol:
            return q, True

        # Keine Verbesserung
        if abs(prev_err - err) < 1e-6:
            return q, False

        prev_err = err

        # ---- Position-Jacobian ----
        eps = 1e-6
        J_pos = np.zeros((3,4))
        for i in range(4):
            dq = q.copy()
            dq[i] += eps

            q_full_eps = np.zeros(16)
            q_full_eps[4:8] = dq

            T_eps = fk(q_full_eps, T_chest_world)["hand_l"]
            pos_eps = T_eps[:3, 3]
            J_pos[:, i] = (pos_eps - pos) / eps

        # ---- DLS Pseudoinverse ----
        lam = 1e-4
        J_pinv = np.linalg.inv(J_pos.T @ J_pos + lam*np.eye(4)) @ J_pos.T

        # ---- Primärbewegung ----
        dq_pos = J_pinv @ e_pos

        # ---- Nullraum-Orientierung: q1+q2+q3 = target_ori_y ----
        e_ori = (q[1] + q[2] + q[3]) - target_ori_y
        grad = np.array([0.0, 1.0, 1.0, 1.0]) * (-e_ori)
        N = np.eye(4) - J_pinv @ J_pos
        dq_ori = N @ grad * alpha

        # ---- Gesamtupdate ----
        q += dq_pos + dq_ori

        # ---- Gelenklimits ----
        q = np.clip(q, q_min, q_max)

        # Limits blockieren & Fehler groß → unlösbar
        touching_limits = np.any((q <= q_min+1e-4) | (q >= q_max-1e-4))
        if touching_limits and err > 0.005:
            return q, False

    return q, False


def ik_right_leg(target_pos, q0, T_chest_world):
    """
    IK für rechtes Bein mit:
    - Positionsconstraint (foot_r_ref -> target_pos)
    - Orientierungsconstraint: Fuß parallel zum Boden
      -> z-Achse des Fußes soll keine x/y-Komponente haben (z_foot_x ≈ 0, z_foot_y ≈ 0)
      Yaw um die Welt-z ist frei.

    target_pos: 3D-Zielposition im Weltkoordinatensystem (np.array shape=(3,))
    q0:         Startgelenkwinkel [hyaw, hpitch, kpitch, ayaw] in rad
    T_chest_world: 4x4 Pose der Brust im Weltkoordinatensystem
    """

    max_iter = 60
    pos_tol = 1e-4       # Toleranz Position
    ori_tol = 1e-3       # Toleranz Orientierung (z_x, z_y)
    lam = 1e-4           # DLS-Dämpfung
    eps = 1e-6           # Schritt für numerische Ableitung

    # yaw, hpitch, kpitch, ayaw – Limits wie zuvor
    q_min = np.deg2rad(np.array([-100, -20,   0, -90]))
    q_max = np.deg2rad(np.array([  -70, 120, 140,  90]))

    # Reichweitenprüfung (optional)
    hip_offset_r = np.array([-0.0434702141, 0.021, -0.0716202141, 1.0])
    hip_world = (T_chest_world @ hip_offset_r)[:3]
    Lmax = 0.3
    if np.linalg.norm(target_pos - hip_world) > Lmax:
        return q0, False

    # Startpose: aus Singulärität raus
    # Wenn du lieber extern ein q0 setzt, kommentier die nächste Zeile aus.
    q0=None
    q = np.deg2rad(np.array([-90, 20, 40, 0])) if q0 is None else q0.copy()

    prev_err = 1e9

    for _ in range(max_iter):
        # ---- FK ----
        q_full = np.zeros(16)
        q_full[12:16] = q
        T_foot = fk(q_full, T_chest_world)["foot_r"]

        pos = T_foot[:3, 3]
        R   = T_foot[:3, :3]

        # Positionsfehler
        e_pos = target_pos - pos             
        err_pos = np.linalg.norm(e_pos)

        # Orientierungsfehler: Fuß-z-Achse soll parallel zu Welt-z sein
        # -> z_foot = R[:,2]; wir wollen z_foot_x = 0, z_foot_y = 0
        z_foot = R[:, 2]
        e_ori = -z_foot[:2] * (err_pos < 0.02)

        err_ori = np.linalg.norm(e_ori)

        # Erfolgskriterium: Position UND Orientierung okay
        if err_pos < pos_tol and err_ori < ori_tol:
            return q, True

        # Abbruch, wenn sich nichts mehr ändert
        total_err = err_pos + err_ori
        if abs(prev_err - total_err) < 1e-7:
            return q, False
        prev_err = total_err

        # ---- Jacobian Position (3x4) ----
        J_pos = np.zeros((3, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += eps

            q_full_eps = np.zeros(16)
            q_full_eps[12:16] = dq
            T_eps = fk(q_full_eps, T_chest_world)["foot_r"]
            pos_eps = T_eps[:3, 3]

            J_pos[:, i] = (pos_eps - pos) / eps
        # --- Yaw-DOFs verstärken ---
        yaw_gain = 1
        J_pos[:,0] *= yaw_gain   # hip_yaw
        J_pos[:,3] *= yaw_gain   # ankle_yaw


        # ---- Jacobian Orientierung (2x4) ----
        # Ableitung von [z_foot_x, z_foot_y] nach q
        J_ori = np.zeros((2, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += eps

            q_full_eps = np.zeros(16)
            q_full_eps[12:16] = dq
            T_eps = fk(q_full_eps, T_chest_world)["foot_r"]
            R_eps = T_eps[:3, :3]
            z_eps = R_eps[:, 2]

            # x,y-Komponenten der Fuß-z-Achse
            z_xy_eps = z_eps[:2]
            z_xy = z_foot[:2]

            J_ori[:, i] = (z_xy_eps - z_xy) / eps

        # ---- Kombinierte IK-Gleichung ----
        # Wir bauen:
        #   [ w_p * J_pos ] dq = [ w_p * e_pos ]
        #   [ w_o * J_ori ]      [ w_o * e_ori ]
        w_pos = 1.0
        # Adaptive Orientierung – aktiviert sich erst wenn wir nahe der Zielposition sind
        if err_pos > 0.2:     # weiter als 2 cm → Orientation aus
            w_ori = 0.0
        else:                  # in 2 cm → Orientation linear aktivieren
            w_ori = 0.5 * (1.0 - err_pos / 0.02)


        J_big = np.vstack((w_pos * J_pos, w_ori * J_ori))     # (5 x 4)
        e_big = np.hstack((w_pos * e_pos, w_ori * e_ori))     # (5,)

        # ---- DLS-Pseudoinverse ----
        JtJ = J_big.T @ J_big
        J_pinv = np.linalg.inv(JtJ + lam * np.eye(4)) @ J_big.T

        dq = J_pinv @ e_big

        # ---- Update ----
        q += dq

        # ---- Gelenklimits ----
        q = np.clip(q, q_min, q_max)

        # Frühabbruch wenn Limits und Fehler groß
        touching_limits = np.any((q <= q_min + 1e-4) | (q >= q_max - 1e-4))
        if touching_limits and err_pos > 0.005:
            return q, False

    return q, False






import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

from fk import fk                    # deine FK-Funktion


def main():
    # ----------------------------
    # 1. MuJoCo Modell laden
    # ----------------------------
    xml_path = (Path(__file__).parent.parent / "Robot" / "Robot_V3.3.xml").resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # set position and orientation of the robot base
    data.qpos[0:3] = [0, -0.3, 0.3]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, np.deg2rad(180)]), 'xyz')
    data.qpos[3:7] = quat
    mujoco.mj_forward(model, data)


    # Brustpose: Roboter befindet sich bei Freijoint = 0
    T_chest_world = np.eye(4)
    T_chest_world[:3, 3] = data.xpos[model.body('chest').id]
    T_chest_world[:3,:3] = data.xmat[model.body('chest').id].reshape(3,3)

    # ----------------------------
    # 2. Zielpunkt definieren
    # ----------------------------
    # Einfacher reachable Testpunkt in der Nähe der Schulter
    target_pos = np.array([0.14, -0.26, 0.12])  
    target_ori_y = np.deg2rad(0)

    # aktueller Winkelzustand (für IK nicht zwingend nötig)
    current_q = np.zeros(4)

    # ----------------------------
    # 3. IK berechnen
    # ----------------------------
    q_ik, success = ik("right_leg", target_pos, target_ori_y, current_q, T_chest_world)
    print("IK Gelenkwinkel:", q_ik)

    # ----------------------------
    # 4. Gelenkwinkel in MuJoCo setzen
    # ----------------------------
    # Reihenfolge wie in deiner FK:
    joint_names = [
        "servo_hip_yaw_r_Revolute_hip_yaw_r",
        "servo_hip_pitch_r_Revolute_hip_pitch_r",
        "servo_knee_pitch_r_Revolute_knee_pitch_r",
        "ankle_link_r_Revolute_ankle_yaw_r"
    ]
    actuator_names = [
        "hip_yaw_r",
        "hip_pitch_r",
        "knee_pitch_r",
        "ankle_yaw_r"
    ]


    
    for i in range(4):
        # qpos setzen
        qid = model.joint(joint_names[i]).qposadr
        data.qpos[qid] = q_ik[i]

        # actuator ctrl setzen (das ist entscheidend!)
        aid = model.actuator(actuator_names[i]).id
        data.ctrl[aid] = q_ik[i]


    mujoco.mj_forward(model, data)

    target_body_id = model.body('target_marker').mocapid
    data.mocap_pos[target_body_id] = target_pos


    # ----------------------------
    # 5. FK separat berechnen (zum Vergleich)
    # ----------------------------
    # globaler Winkelvektor q für dein FK (Größe 16)
    q_full = np.zeros(16)
    q_full[12:16] = q_ik  # rechter Fuß

    fk_res = fk(q_full, T_chest_world)
    fk_hand = fk_res["foot_r"][:3, 3]

    # ----------------------------
    # 6. MuJoCo Endeffektorposition aus Simulation
    # ----------------------------
    # In deiner XML ist der Referenzpunkt "foot_r_ref"
    body_id = model.body("foot_r_ref").id
    sim_pos = data.xpos[body_id]

    # ----------------------------
    # 7. Ergebnisse ausgeben
    # ----------------------------
    print("\n---- Vergleich Fußposition ----")
    print("Zielposition:       ", target_pos)
    print("FK Ergebnis:        ", fk_hand)
    print("MuJoCo Simulation:  ", sim_pos)
    print("Fehler FK:          ", np.linalg.norm(fk_hand - target_pos))
    print("Fehler MuJoCo:      ", np.linalg.norm(sim_pos - target_pos))
    print("IK Erfolg:", success)
    # ----------------------------
    # 8. Viewer öffnen (optional)
    # ----------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, dt - (time.time() - start)))


if __name__ == "__main__":
    main()
