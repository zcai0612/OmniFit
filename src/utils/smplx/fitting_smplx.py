import trimesh
import json
from smplx import SMPLX
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os
import theseus as th
from src.utils.smplx.lmks import get_smplx_landmarks_from_file

def fit_smplx(
    smplx_model,
    pred_lmks,
    lmk_json_file,
    scale=1.0,
    steps_stage0=20,
    steps_stage1=20,
    steps_stage2=60,
    lr_stage0=5e-1,
    lr_stage1=5e-1,
    lr_stage2=2e-1,
):
    '''
        Fit SMPL-X model from the predicted landmarks.

        args:
            pred_lmks: (B, N_lmk, 3) predicted landmarks

    '''

    B, L = pred_lmks.shape[0], pred_lmks.shape[1]
    device = pred_lmks.device

    lmk_indices = get_smplx_landmarks_from_file(lmk_json_file)

    loss_weights = {
        "lmk_loss": 1.0,
    }

    def lmk_error_fn_0(optim_vars, aux_vars):
        (
            body_pose,
            lhand_pose,
            rhand_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
            expression_optimized,
            shape_optimized,
            global_orient,
            translation,
        ) = optim_vars
        pred_lmks_position = aux_vars[0]

        batch_size = shape_optimized.tensor.shape[0]
        shape_frozen = torch.zeros((batch_size, 10 - 2)).to(device)
        expression_frozen = torch.zeros((batch_size, 10 - 2)).to(device)

        # forward
        smplx_output = smplx_model(
            global_orient=global_orient.tensor,
            body_pose=body_pose.tensor,
            betas=torch.cat([shape_optimized.tensor, shape_frozen], dim=1),
            expression=torch.cat(
                [expression_optimized.tensor, expression_frozen], dim=1
            ),
            jaw_pose=jaw_pose.tensor,
            leye_pose=leye_pose.tensor,
            reye_pose=reye_pose.tensor,
            left_hand_pose=lhand_pose.tensor,
            right_hand_pose=rhand_pose.tensor,
            transl=translation.tensor,
            return_verts=True,
        )
        smplx_vertices = smplx_output.vertices  # shape(B, V, 3)

        lmk_vindices = (
            torch.tensor(lmk_indices, device=smplx_vertices.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        forwarded_lmks_position = torch.gather(
            smplx_vertices, 1, lmk_vindices.unsqueeze(-1).expand(-1, -1, 3)
        )  # shape(B, num_lmks, 3)

        err = pred_lmks_position.tensor - forwarded_lmks_position
        err = err.reshape(batch_size, -1)
        # print(err.shape)

        return err

    def lmk_error_fn_1(optim_vars, aux_vars):
        (
            body_pose,
            lhand_pose,
            rhand_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
            expression,
            shape,
            global_orient,
            translation,
        ) = optim_vars
        pred_lmks_position = aux_vars[0]

        batch_size = shape.tensor.shape[0]

        # forward
        smplx_output = smplx_model(
            global_orient=global_orient.tensor,
            body_pose=body_pose.tensor,
            betas=shape.tensor,
            expression=expression.tensor,
            jaw_pose=jaw_pose.tensor,
            leye_pose=leye_pose.tensor,
            reye_pose=reye_pose.tensor,
            left_hand_pose=lhand_pose.tensor,
            right_hand_pose=rhand_pose.tensor,
            transl=translation.tensor,
            return_verts=True,
        )
        smplx_vertices = smplx_output.vertices  # shape(B, V, 3)

        lmk_vindices = (
            torch.tensor(lmk_indices, device=smplx_vertices.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        forwarded_lmks_position = torch.gather(
            smplx_vertices, 1, lmk_vindices.unsqueeze(-1).expand(-1, -1, 3)
        )  # shape(B, num_lmks, 3)

        err = pred_lmks_position.tensor - forwarded_lmks_position
        err = err.reshape(batch_size, -1)

        return err

    def lmk_error_fn_stage0(optim_vars, aux_vars):
        """Error function for stage 0: only optimize global_orient and translation"""
        global_orient, translation = optim_vars
        pred_lmks_position = aux_vars[0]

        batch_size = global_orient.tensor.shape[0]
        
        # Use default/zero values for all other parameters
        body_pose = torch.zeros((batch_size, 63)).to(device)
        lhand_pose = torch.zeros((batch_size, 45)).to(device)
        rhand_pose = torch.zeros((batch_size, 45)).to(device)
        jaw_pose = torch.zeros((batch_size, 3)).to(device)
        leye_pose = torch.zeros((batch_size, 3)).to(device)
        reye_pose = torch.zeros((batch_size, 3)).to(device)
        shape = torch.zeros((batch_size, 10)).to(device)
        expression = torch.zeros((batch_size, 10)).to(device)

        # forward
        smplx_output = smplx_model(
            global_orient=global_orient.tensor,
            body_pose=body_pose,
            betas=shape,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            transl=translation.tensor,
            return_verts=True,
        )
        smplx_vertices = smplx_output.vertices  # shape(B, V, 3)

        lmk_vindices = (
            torch.tensor(lmk_indices, device=smplx_vertices.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        forwarded_lmks_position = torch.gather(
            smplx_vertices, 1, lmk_vindices.unsqueeze(-1).expand(-1, -1, 3)
        )  # shape(B, num_lmks, 3)

        err = pred_lmks_position.tensor - forwarded_lmks_position
        err = err.reshape(batch_size, -1)

        return err

    # STAGE 0: ONLY OPTIMIZE GLOBAL_ORIENT AND TRANSLATION
    # print("Optimization stage 0:")

    B = pred_lmks.shape[0]
    pred_lmks_position = th.Variable(
        tensor=pred_lmks, name="pred_lmks_position"
    )

    # Initialize only global_orient and translation for optimization
    global_orient = torch.zeros((B, 3)).to(device)
    translation = torch.zeros((B, 3)).to(device)

    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")

    optim_vars = [global_orient, translation]
    aux_vars = [pred_lmks_position]

    w_lmk = th.ScaleCostWeight(loss_weights["lmk_loss"])
    lmk_cost_function = th.AutoDiffCostFunction(
        optim_vars,
        lmk_error_fn_stage0,
        len(lmk_indices) * 3,
        cost_weight=w_lmk,
        aux_vars=aux_vars,
        name="lmk_cost_function",
    )

    objective = th.Objective().to(device)
    objective.add(lmk_cost_function)
    optimizer = th.LevenbergMarquardt(
        objective, max_iterations=steps_stage2, step_size=lr_stage2
    )
    theseus_layer = th.TheseusLayer(optimizer).to(device)

    theseus_inputs = {
        "global_orient": global_orient,
        "translation": translation,
        "pred_lmks_position": pred_lmks_position,
    }

    updated_inputs, _ = theseus_layer.forward(
        theseus_inputs, optimizer_kwargs={"verbose": False}
    )

    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]

    # STAGE 1: ONLY OPTIMIZE POSE AND TOP BETAS (previously stage 0)
    # print("Optimization stage 1:")

    # Initialize optimization variables, using results from stage 0
    body_pose = torch.zeros((B, smplx_model.NUM_BODY_JOINTS * 3)).to(
        device
    )
    lhand_pose = torch.zeros((B, smplx_model.NUM_HAND_JOINTS * 3)).to(
        device
    )
    rhand_pose = torch.zeros((B, smplx_model.NUM_HAND_JOINTS * 3)).to(
        device
    )
    jaw_pose = torch.zeros((B, 1 * 3)).to(device)
    leye_pose = torch.zeros((B, 1 * 3)).to(device)
    reye_pose = torch.zeros((B, 1 * 3)).to(device)
    expression_optimized = torch.zeros((B, 2)).to(device)
    shape_optimized = torch.zeros((B, 2)).to(device)
    # Use global_orient and translation from stage 0
    global_orient = global_orient.detach()
    translation = translation.detach()

    body_pose = th.Vector(tensor=body_pose, name="body_pose")
    lhand_pose = th.Vector(tensor=lhand_pose, name="lhand_pose")
    rhand_pose = th.Vector(tensor=rhand_pose, name="rhand_pose")
    jaw_pose = th.Vector(tensor=jaw_pose, name="jaw_pose")
    leye_pose = th.Vector(tensor=leye_pose, name="leye_pose")
    reye_pose = th.Vector(tensor=reye_pose, name="reye_pose")
    expression_optimized = th.Vector(
        tensor=expression_optimized, name="expression_optimized"
    )
    shape_optimized = th.Vector(tensor=shape_optimized, name="shape_optimized")
    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")

    pred_lmks_position = th.Variable(
        tensor=pred_lmks, name="pred_lmks_position"
    )

    optim_vars = [
        body_pose,
        lhand_pose,
        rhand_pose,
        jaw_pose,
        leye_pose,
        reye_pose,
        expression_optimized,
        shape_optimized,
        global_orient,
        translation,
    ]
    aux_vars = [pred_lmks_position]

    w_lmk = th.ScaleCostWeight(loss_weights["lmk_loss"])
    # monitor_memory()
    lmk_cost_function = th.AutoDiffCostFunction(
        optim_vars,
        lmk_error_fn_0,
        len(lmk_indices) * 3,
        cost_weight=w_lmk,
        aux_vars=aux_vars,
        name="lmk_cost_function",
    )
    # monitor_memory()
    objective = th.Objective().to(device)
    objective.add(lmk_cost_function)

    optimizer = th.LevenbergMarquardt(
        objective, max_iterations=steps_stage0, step_size=lr_stage0
    )
    # monitor_memory()
    # optimizer = th.GaussNewton(objective, max_iterations=steps_stage0, step_size=lr_stage0)
    theseus_layer = th.TheseusLayer(optimizer).to(device)

    theseus_inputs = {
        "body_pose": body_pose,
        "lhand_pose": lhand_pose,
        "rhand_pose": rhand_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "expression_optimized": expression_optimized,
        "shape_optimized": shape_optimized,
        "global_orient": global_orient,
        "translation": translation,
        "pred_lmks_position": pred_lmks_position,
    }
    # monitor_memory()

    updated_inputs, _ = theseus_layer.forward(
        theseus_inputs, optimizer_kwargs={"verbose": False, "damping": 0.01}
    )  # TODO: damping = ??

    body_pose = updated_inputs["body_pose"]
    lhand_pose = updated_inputs["lhand_pose"]
    rhand_pose = updated_inputs["rhand_pose"]
    jaw_pose = updated_inputs["jaw_pose"]
    leye_pose = updated_inputs["leye_pose"]
    reye_pose = updated_inputs["reye_pose"]
    expression_optimized = updated_inputs["expression_optimized"]
    shape_optimized = updated_inputs["shape_optimized"]
    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]

    # STAGE 2: OPTIMIZE POSE AND ALL BETAS (previously stage 1)
    # print("Optimization stage 2:")

    body_pose = body_pose.detach()
    lhand_pose = lhand_pose.detach()
    rhand_pose = rhand_pose.detach()
    jaw_pose = jaw_pose.detach()
    leye_pose = leye_pose.detach()
    reye_pose = reye_pose.detach()
    shape_frozen = torch.zeros((B, 10 - 2)).to(device)
    expression_frozen = torch.zeros((B, 10 - 2)).to(device)
    shape = torch.cat([shape_optimized, shape_frozen], dim=1).detach()
    expression = torch.cat([expression_optimized, expression_frozen], dim=1).detach()
    global_orient = global_orient.detach()
    translation = translation.detach()

    body_pose = th.Vector(tensor=body_pose, name="body_pose")
    lhand_pose = th.Vector(tensor=lhand_pose, name="lhand_pose")
    rhand_pose = th.Vector(tensor=rhand_pose, name="rhand_pose")
    jaw_pose = th.Vector(tensor=jaw_pose, name="jaw_pose")
    leye_pose = th.Vector(tensor=leye_pose, name="leye_pose")
    reye_pose = th.Vector(tensor=reye_pose, name="reye_pose")
    expression = th.Vector(tensor=expression, name="expression")
    shape = th.Vector(tensor=shape, name="shape")
    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")

    optim_vars = [
        body_pose,
        lhand_pose,
        rhand_pose,
        jaw_pose,
        leye_pose,
        reye_pose,
        expression,
        shape,
        global_orient,
        translation,
    ]
    aux_vars = [pred_lmks_position]

    w_lmk = th.ScaleCostWeight(loss_weights["lmk_loss"])
    lmk_cost_function = th.AutoDiffCostFunction(
        optim_vars,
        lmk_error_fn_1,
        len(lmk_indices) * 3,
        cost_weight=w_lmk,
        aux_vars=aux_vars,
        name="lmk_cost_function",
    )

    objective = th.Objective().to(device)
    objective.add(lmk_cost_function)
    optimizer = th.LevenbergMarquardt(
        objective, max_iterations=steps_stage1, step_size=lr_stage1
    )
    theseus_layer = th.TheseusLayer(optimizer).to(device)

    theseus_inputs = {
        "body_pose": body_pose,
        "lhand_pose": lhand_pose,
        "rhand_pose": rhand_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "expression": expression,
        "shape": shape,
        "global_orient": global_orient,
        "translation": translation,
        "pred_lmks_position": pred_lmks_position,
    }

    updated_inputs, _ = theseus_layer.forward(
        theseus_inputs, optimizer_kwargs={"verbose": False}
    )  # TODO: damping = ??

    body_pose = updated_inputs["body_pose"]
    lhand_pose = updated_inputs["lhand_pose"]
    rhand_pose = updated_inputs["rhand_pose"]
    jaw_pose = updated_inputs["jaw_pose"]
    leye_pose = updated_inputs["leye_pose"]
    reye_pose = updated_inputs["reye_pose"]
    expression = updated_inputs["expression"]
    shape = updated_inputs["shape"]
    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]
    # print(translation)
    # exit()

    # get final smpl meshes
    smplx_output = smplx_model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=shape,
        expression=expression,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=lhand_pose,
        right_hand_pose=rhand_pose,
        transl=translation,
        return_verts=True,
    )
    joints = smplx_output.joints  # shape(B, J, 3)

    final_mesh_list = []
    for b in range(B):
        final_smpl_mesh = trimesh.Trimesh(
            smplx_output.vertices[b].detach().cpu().numpy(),
            smplx_model.faces,
            process=False,
            maintain_order=True,
        )
        final_mesh_list.append(final_smpl_mesh)


    output_smplx_info = {}
    output_smplx_info["global_orient"] = torch.nn.Parameter(global_orient)
    output_smplx_info["body_pose"] = torch.nn.Parameter(body_pose)
    output_smplx_info["lhand_pose"] = torch.nn.Parameter(lhand_pose)
    output_smplx_info["rhand_pose"] = torch.nn.Parameter(rhand_pose)
    output_smplx_info["jaw_pose"] = torch.nn.Parameter(jaw_pose)
    output_smplx_info["leye_pose"] = torch.nn.Parameter(leye_pose)
    output_smplx_info["reye_pose"] = torch.nn.Parameter(reye_pose)
    # output_smplx_info["pose"] = torch.nn.Parameter(
    #     torch.cat(
    #         [
    #             global_orient,
    #             body_pose,
    #             jaw_pose,
    #             leye_pose,
    #             reye_pose,
    #             lhand_pose,
    #             rhand_pose,
    #         ],
    #         dim=1,
    #     )
    # )
    output_smplx_info["beta"] = torch.nn.Parameter(shape)
    output_smplx_info["expression"] = torch.nn.Parameter(expression)
    # output_smpl_info["global_orient"] = global_orient.detach().cpu().numpy()
    output_smplx_info["trans"] = torch.nn.Parameter(translation)
    output_smplx_info["joints"] = joints
    output_smplx_info["scale"] = torch.tensor([scale])
    # shape(B, 23, 3), shape(B, 10), shape(B, 3), shape(B, 3), shape(B, 45, 3)

    return (
        final_mesh_list,
        # pred_lmks_position.tensor,
        output_smplx_info,
    )


# if __name__ == "__main__":
#     test_smplx_mesh_path = "data/mesh_smplx_1_0120_335.obj"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     from src.mesh_utils.mesh_util import load_obj_mesh, normalize_vertices
#     v, f = load_obj_mesh(test_smplx_mesh_path)
#     v = normalize_vertices(v, bound=0.9)
#     lmk_indices = get_lmk_indices()
#     lmk = v[lmk_indices, :]
#     lmk = torch.from_numpy(lmk).float().unsqueeze(0).to(device)

#     smplx_model = SMPLX(
#         model_path="human_models/models/smplx/SMPLX_NEUTRAL.npz",
#         use_pca=False,
#         flat_hand_mean=True,
#     ).to(device)

#     final_mesh_list, _ = fit_smplx(
#         smplx_model,
#         lmk,
#     )

#     for i, mesh in enumerate(final_mesh_list):
#         mesh.export(f"fitted_smplx_{i}.obj")


        