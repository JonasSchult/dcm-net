"""General purpose visualizer for semantic segmentation results on various datasets super-fueled by open3D.
"""
import os
import open3d
import torch
import numpy as np
from termcolor import colored
from base.base_dataset import BaseDataSet


class SemSegVisualizer:
    """Visualize meshes from various datasets with open3D. Key Events show RGB, ground truth, prediction or differences."""

    def __init__(self, dataset: BaseDataSet, save_dir: str = ""):
        """Initialize Semantic Segmentation Visualizer which shows meshes with optional prediction and ground truth

        Arguments:
            dataset  {BaseDataSet} -- Examples from which dataset we want to visualize
            save_dir {str}         -- Directory in which .ply files should be saved
        """

        # keep a pointer to the dataset to retrieve relevant information, such as color mapping
        self._dataset = dataset
        
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        assert os.path.isdir(save_dir) or save_dir == ""
        self._save_dir = save_dir

    def visualize_result(self, mesh_name, prediction=None, gt=None):
        mesh = self._dataset.get_mesh(mesh_name)
        mesh.compute_vertex_normals()

        vis = open3d.VisualizerWithKeyCallback()
        vis.create_window(width=1600, height=1200)

        # PREPARE RGB COLOR SWITCH
        rgb_colors = open3d.Vector3dVector(np.asarray(mesh.vertex_colors))

        def colorize_rgb(visu):
            mesh.vertex_colors = rgb_colors
            visu.update_geometry()
            visu.update_renderer()

        vis.register_key_callback(ord('H'), colorize_rgb)

        if type(prediction) == torch.Tensor:
            # PREPARE PREDICTION COLOR SWITCH
            pred_colors = open3d.Vector3dVector(
                self._dataset.color_map[prediction.long()] / 255.)

            def colorize_pred(visu):
                mesh.vertex_colors = pred_colors
                visu.update_geometry()
                visu.update_renderer()

            vis.register_key_callback(ord('J'), colorize_pred)

        if type(gt) == torch.Tensor:
            # PREPARE GROUND TRUTH COLOR SWITCH
            gt_colors = open3d.Vector3dVector(self._dataset.color_map[gt.long()] / 255.)

            def colorize_gt(visu):
                mesh.vertex_colors = gt_colors
                visu.update_geometry()
                visu.update_renderer()

            vis.register_key_callback(ord('K'), colorize_gt)

        if type(gt) == torch.Tensor and type(prediction) == torch.Tensor:
            # PREAPRE DIFFERENCE COLOR SWITCH
            pos = (prediction == gt)
            neg = ((prediction != gt) & (gt != 0)) * 2

            differences = pos + neg
            diff_colors = open3d.Vector3dVector(
                self._dataset.pos_neg_map[differences.long()] / 255.)

            def colorize_diff(visu):
                mesh.vertex_colors = diff_colors
                visu.update_geometry()
                visu.update_renderer()

            vis.register_key_callback(ord('F'), colorize_diff)

        def save_room(visu):
            mesh.vertex_colors = rgb_colors
            open3d.io.write_triangle_mesh(
                f"{self._save_dir}/SemSegVisualizer_rgb.ply", mesh)

            if type(prediction) == torch.Tensor:
                mesh.vertex_colors = pred_colors
                open3d.io.write_triangle_mesh(
                    f"{self._save_dir}/SemSegVisualizer_pred.ply", mesh)

            if type(gt) == torch.Tensor:
                mesh.vertex_colors = gt_colors
                open3d.io.write_triangle_mesh(
                    f"{self._save_dir}/SemSegVisualizer_gt.ply", mesh)

            if type(gt) == torch.Tensor and type(prediction) == torch.Tensor:
                mesh.vertex_colors = diff_colors
                open3d.io.write_triangle_mesh(
                    f"{self._save_dir}/SemSegVisualizer_diff.ply", mesh)
            print(colored(
                f"PLY meshes successfully stored in {os.path.abspath(self._save_dir)}", 'green'))

        vis.register_key_callback(ord('D'), save_room)
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()
