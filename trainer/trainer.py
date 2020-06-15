from base import BaseTrainer
from metric.metrics import *
from metric.confusionmatrix import *
import time
import os
from tqdm import tqdm
from sklearn.neighbors import BallTree
import dataset
from utils.SemSegVisualizer import SemSegVisualizer


def _exit(writer, id):
    writer.close()
    exit(id)


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, optimizer, resume, config, lr_scheduler,
                 data_loader, valid_data_loader=None, train_logger=None,
                 eval_mode=False, is_runtime=False, max_points=None, max_points_epoch=None, vis=False, n_gpu=1,
                 s3dis_gt_pcd=False, not_save=False):
        self.num = 0
        super(Trainer, self).__init__(model, loss, optimizer,
                                      lr_scheduler, resume, config, train_logger)
        self.loss.device = self.device
        self._valid_iterations = config['trainer']['iterations']
        self.config = config
        self.data_loader = data_loader
        self._is_evaluation = eval_mode
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size)
                            ) if data_loader is not None else 1
        self._resume_path = resume
        self._is_runtime = is_runtime
        self._max_points = max_points
        self._max_points_epoch = max_points_epoch
        self._vis = vis
        self._n_gpu = n_gpu
        self._s3dis_gt_pcd = s3dis_gt_pcd
        self._is_scannet = not type(self.data_loader.dataset) is dataset.s3dis.S3DIS if data_loader is not None else not type(
            self.valid_data_loader.dataset) is dataset.s3dis.S3DIS
        self._not_save = not_save

        output_dir = config['trainer'].get('output_dir', False)

        if output_dir:
            self._output_dir = f"{config['trainer']['output_dir']}/{config['name']}"
        else:
            self._output_dir = False

        self._benchmark = config['valid_dataset']['args']['benchmark']

    def _eval_metrics(self, output, target):
        acc_metrics = []
        for i, metric in enumerate(self.metrics):
            acc_metrics.append(metric(output, target))

            for j in range(len(acc_metrics[-1])):
                self.writer.add_scalar(
                    f'{metric.__name__}_{j}', acc_metrics[-1][j])

        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        if 'set_current_epoch' in dir(self.loss):
            self.loss.set_current_epoch(epoch)
        if self._max_points and self._max_points_epoch:
            epoch_delta = epoch - self._max_points_epoch
            self._max_points = int(self._max_points * 1.05**epoch_delta)
            print('current max points:', self._max_points)

        if self._vis:
            self._visualize_epoch()
            exit(12)
        if self._is_runtime:
            runtimes = self._runtime_epoch()

            with open(f"runtime_results/{self.config['name']}_merged_runtimes.csv", 'w') as csv_file:
                for mesh_name, inference_time in runtimes.items():
                    csv_file.write(f"{mesh_name};{inference_time}\n")

            _exit(self.writer, 0)

        if self._is_evaluation:
            val_log = self._valid_epoch(epoch)
            log = {**val_log}
            print(log)

            eval_file = self._resume_path.replace('.pth', '.txt')

            if not self._not_save:
                with open(eval_file, 'w') as outfile:
                    outfile.write(str(val_log))

            _exit(self.writer, 10)
        else:
            try:
                self.model.train()

                conf_matrix = ConfusionMatrix(
                    len(self.data_loader.dataset.classes))
                iou = IoU(ignore_index=self.data_loader.dataset.ignore_classes)

                ckpt_timer = time.time()
                total_loss = 0
                for batch_idx, sample in enumerate(self.data_loader):
                    if self.num != 0:
                        if self.num + batch_idx * self.data_loader.batch_size >= self.data_loader.n_samples():
                            self.num = 0
                            break

                    if self._max_points:
                        if self._n_gpu > 1:
                            num_nodes = sum(
                                [data.num_nodes for data in sample])
                        else:
                            num_nodes = sample.num_nodes

                        if self._max_points < num_nodes:
                            self.logger.info(f"Batch with {num_nodes} points is bigger than "
                                             f"max number of points ({self._max_points})")
                            continue

                    if time.time() - ckpt_timer > 10*60:
                        print('save timed checkpoint')
                        self._save_checkpoint(
                            epoch, emergency=True, num=self.num + batch_idx * self.data_loader.batch_size)
                        ckpt_timer = time.time()

                    if self._n_gpu <= 1:
                        sample.to(self.device)

                    self.optimizer.zero_grad()

                    try:
                        output = self.model(sample)
                        if self._n_gpu > 1:
                            y = torch.cat([data.y for data in sample]).to(
                                output.device)
                        else:
                            y = sample.y

                        loss = self.loss(output, y)

                        loss.backward()
                        self.optimizer.step()
                    except RuntimeError as e:
                        print(str(e))
                        if 'out of memory' in str(e):
                            print(
                                '| WARNING: ran out of memory, store model and restart process')
                            self._save_checkpoint(
                                epoch, emergency=True, num=self.num + batch_idx * self.data_loader.batch_size)

                            if self._n_gpu > 1:
                                num_nodes = sum(
                                    [data.num_nodes for data in sample])
                            else:
                                num_nodes = sample.num_nodes

                            if self._max_points:
                                if num_nodes < self._max_points:
                                    max_points_file_path = os.path.join(
                                        self.checkpoint_dir, 'max_num_points.txt')
                                    with open(max_points_file_path, 'w') as points_file:
                                        points_file.write(
                                            f"{epoch};{num_nodes}")
                            else:
                                max_points_file_path = os.path.join(
                                    self.checkpoint_dir, 'max_num_points.txt')
                                with open(max_points_file_path, 'w') as points_file:
                                    points_file.write(f"{epoch};{num_nodes}")

                            _exit(self.writer, 91)
                        _exit(self.writer, 1)

                    total_loss += loss.item()

                    conf_matrix.add(output, y)

                    if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                        self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                            epoch,
                            self.num + batch_idx * self.data_loader.batch_size,
                            self.data_loader.n_samples(),
                            100.0 * (self.num / self.data_loader.batch_size + batch_idx) / len(
                                self.data_loader),
                            loss.item()))

                metrics = iou.value(conf_matrix.value(normalized=False))
                self.writer.add_scalar(
                    'train/lr', self.optimizer.param_groups[0]['lr'], global_step=epoch)
                self.writer.add_scalar(
                    'train/loss', total_loss, global_step=epoch)
                self.writer.add_scalar(
                    'train/mIoU', metrics['mean_iou'], global_step=epoch)
                self.writer.add_scalar(
                    'train/mPrec', metrics['mean_precision'], global_step=epoch)
                self.writer.add_scalar(
                    'train/oPrec', metrics['overall_precision'], global_step=epoch)

                log = {
                    'loss': total_loss / len(self.data_loader),
                    'mean IoU': metrics['mean_iou'],
                    'mean precision': metrics['mean_precision'],
                    'overall precision': metrics['overall_precision']
                }

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                return log
            except KeyboardInterrupt as e:
                print(str(e))
                self._save_checkpoint(epoch, emergency=True)
                _exit(self.writer, 1)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        if 'set_current_epoch' in dir(self.loss):
            self.loss.set_current_epoch(-1)
        self.model.eval()
        total_val_loss = 0

        conf_matrix = ConfusionMatrix(
            len(self.valid_data_loader.dataset.classes))
        iou = IoU(ignore_index=self.valid_data_loader.dataset.ignore_classes)

        full_predictions = {}

        full_mesh_labels = {}

        from collections import defaultdict
        runtimes = defaultdict(float)

        print(self._valid_iterations)
        for round_num in range(self._valid_iterations):
            print(f"CURRENT ROUND: {round_num}")
            with torch.no_grad():
                for batch_idx, sample in tqdm(enumerate(self.valid_data_loader)):
                    if self._n_gpu <= 1:
                        sample.to(self.device)
                    try:
                        if self._n_gpu <= 1:
                            start = time.time()
                            output = self.model(sample)
                            end = time.time() - start
                            runtimes[sample.x.shape[0]] += end
                        else:
                            output = self.model(sample, True)

                        if self._n_gpu > 1:
                            sample = output[1]
                            output = output[0]

                    except RuntimeError as e:
                        print(str(e))
                        if 'out of memory' in str(e):
                            print('| EVAL OUT OF MEMORY')
                            print(sample.name)
                            _exit(self.writer, 1)
                        _exit(self.writer, 1)

                    class_predictions = torch.argmax(output, 1)
                    if not self._s3dis_gt_pcd:
                        if type(sample.original_index_traces) != list:
                            class_predictions_block = class_predictions[sample.original_index_traces]
                            batch_block = sample.batch[sample.original_index_traces]
                        else:
                            assert len(sample.original_index_traces) == 1
                            class_predictions_block = class_predictions
                            batch_block = sample.batch

                            for i in reversed(range(len(sample.original_index_traces[0]))):
                                class_predictions_block = class_predictions_block[
                                    sample.original_index_traces[0][i]]
                                batch_block = batch_block[sample.original_index_traces[0][i]]
                    else:
                        class_predictions_block = class_predictions
                        batch_block = sample.batch

                    for i in range(len(sample.name)):
                        mesh_name = sample.name[i]
                        tmp_name = mesh_name

                        if not self._is_scannet:
                            if mesh_name in [
                                'Area_5_hallway_1_0.pt',
                                'Area_5_hallway_1_1.pt',
                                'Area_5_hallway_2_0.pt',
                                'Area_5_hallway_2_1.pt']:

                                tmp_name = mesh_name.rsplit('_',1)
                                tmp_name.pop(1)
                                tmp_name = '_'.join(tmp_name)
                                tmp_name = tmp_name + ".pt"

                            gt_pcd = self.valid_data_loader.dataset.get_gt_pointcloud(tmp_name)

                        if tmp_name not in full_predictions:
                            if not self._benchmark:
                                if self._is_scannet:
                                    # SCANNET
                                    full_mesh_labels[mesh_name] = sample.y
                                    full_predictions[mesh_name] = torch.zeros((full_mesh_labels[mesh_name].shape[0],
                                                                               len(self.valid_data_loader.dataset.classes)),
                                                                              dtype=torch.long)
                                else:
                                    # S3DIS
                                    full_predictions[tmp_name] = torch.zeros((gt_pcd.shape[0],
                                        len(self.valid_data_loader.dataset.classes)),
                                        dtype=torch.uint8)
                            else:
                                # SCANNET BENCHMARK
                                m = mesh_name.split('.')[0]
                                full_predictions[mesh_name] = torch.zeros((
                                    self.valid_data_loader.dataset.get_full_mesh_size(
                                        m),
                                    len(self.valid_data_loader.dataset.classes)), dtype=torch.long)

                        if self._is_scannet:
                            full_predictions[mesh_name][
                                torch.arange(
                                    0, class_predictions_block[batch_block == i].shape[0], dtype=torch.long),
                                class_predictions_block[batch_block == i]] += 1
                        else:
                            full_mesh_labels[tmp_name] = gt_pcd[:, -1]

                            level_0_coords = self.valid_data_loader.dataset.get_level_0(
                                mesh_name)

                            ball_tree = BallTree(
                                level_0_coords[:, :3])
                            dist, ind = ball_tree.query(
                                gt_pcd[:, :3], k=1)

                            torch_ind = torch.from_numpy(
                                ind.flatten())
                            torch_dist = torch.from_numpy(
                                dist.flatten())

                            class_predictions_block = class_predictions[torch_ind]
                            batch_block = sample.batch[torch_ind]

                            full_predictions[tmp_name][torch_dist.flatten() < 0.12,
                                                        class_predictions_block[torch_dist.flatten() < 0.12]] += 1
                        
                    if not self._benchmark:
                        if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                            self.logger.info('Valid Epoch: {} [{}/{} ({:.0f}%)]'.format(
                                epoch,
                                batch_idx * self.valid_data_loader.batch_size,
                                self.valid_data_loader.n_samples(),
                                100.0 * batch_idx / len(self.valid_data_loader)))
                    else:
                        if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                            self.logger.info('Valid Epoch: {} [{}/{} ({:.0f}%)] '.format(
                                epoch,
                                batch_idx * self.valid_data_loader.batch_size,
                                self.valid_data_loader.n_samples(),
                                100.0 * batch_idx / len(self.valid_data_loader)))

        if self._output_dir:
            if not os.path.exists(f"{self._output_dir}/predictions"):
                os.makedirs(f"{self._output_dir}/predictions")

            if not self._benchmark:
                if not os.path.exists(f"{self._output_dir}/gt"):
                    os.makedirs(f"{self._output_dir}/gt")

        for key in sorted(full_predictions):
            if not self._benchmark:
                conf_matrix.add(full_predictions[key], full_mesh_labels[key])

            if self._output_dir:
                print(full_predictions[key])
                _, predicted_class = full_predictions[key].max(1)
                predicted_class = predicted_class.view(-1)

                if torch.is_tensor(predicted_class):
                    predicted_class = predicted_class.cpu().numpy()

                remap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
                remapped_prediction = np.asarray(
                    [remap[int(s)] for s in predicted_class])

                if not self._benchmark:
                    gt_class = full_mesh_labels[key].cpu().numpy()
                    remapped_gt = np.asarray([remap[int(s)] for s in gt_class])
                    with open(f"{self._output_dir}/gt/{key}.txt", 'w') as gt_file:
                        for i in range(len(remapped_prediction)):
                            gt_file.write(f"{remapped_gt[i]}\n")

                k = key.split('.')[0]
                with open(f"{self._output_dir}/predictions/{k}.txt", 'w') as pred_file:
                    for i in range(len(remapped_prediction)):
                        pred_file.write(f"{remapped_prediction[i]}\n")

        if not self._benchmark:
            metrics = iou.value(conf_matrix.value(normalized=False))
            self.writer.add_scalar(
                'val/loss', total_val_loss, global_step=epoch)
            self.writer.add_scalar(
                'val/mIoU', metrics['mean_iou'], global_step=epoch)
            self.writer.add_scalar(
                'val/mPrec', metrics['mean_precision'], global_step=epoch)
            self.writer.add_scalar(
                'val/oPrec', metrics['overall_precision'], global_step=epoch)

            print('validation IoU per class')
            print('=============')
            print(metrics['iou'])

            print('validation precision per class')
            print('===================')
            print(metrics['precision_per_class'])

            log = {
                'val mean IoU': metrics['mean_iou'],
                'val mean precision': metrics['mean_precision'],
                'val overall precision': metrics['overall_precision'],
            }

        else:
            log = 'finished'

        return log

    def _visualize_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        visualizer = SemSegVisualizer(
            self.valid_data_loader.dataset, "visualizations/")

        self.model.eval()
        full_predictions = {}
        full_mesh_labels = {}
        for round_num in range(self._valid_iterations):
            with torch.no_grad():
                for batch_idx, sample in enumerate(self.valid_data_loader):
                    sample.to(self.device)
                    try:
                        output = self.model(sample)
                    except RuntimeError as e:
                        print(str(e))
                        if 'out of memory' in str(e):
                            print('| EVAL OUT OF MEMORY')
                            _exit(self.writer, 1)
                        _exit(self.writer, 1)

                    class_predictions = torch.argmax(output, 1)

                    if not self._s3dis_gt_pcd:
                        if type(sample.original_index_traces) != list:
                            class_predictions_block = class_predictions[sample.original_index_traces]
                            batch_block = sample.batch[sample.original_index_traces]
                        else:
                            assert len(sample.original_index_traces) == 1
                            class_predictions_block = class_predictions
                            batch_block = sample.batch

                            for i in reversed(range(len(sample.original_index_traces[0]))):
                                class_predictions_block = class_predictions_block[
                                    sample.original_index_traces[0][i]]
                                batch_block = batch_block[sample.original_index_traces[0][i]]
                    else:
                        class_predictions_block = class_predictions
                        batch_block = sample.batch

                    for i in range(len(sample.name)):
                        mesh_name = sample.name[i]
                        if mesh_name not in full_predictions:
                            if not self._benchmark:
                                if self._is_scannet:
                                    # SCANNET
                                    full_mesh_labels[mesh_name] = sample.y
                                    full_predictions[mesh_name] = torch.zeros((full_mesh_labels[mesh_name].shape[0],
                                                                               len(
                                                                                   self.valid_data_loader.dataset.classes)),
                                                                              dtype=torch.long)
                                else:
                                    # S3DIS
                                    if self._s3dis_gt_pcd:
                                        if mesh_name in [
                                            'Area_5_hallway_1_0.pt',
                                            'Area_5_hallway_1_1.pt',
                                            'Area_5_hallway_2_0.pt',
                                                'Area_5_hallway_2_1.pt']:
                                            gt_pcd = self.valid_data_loader.dataset.get_gt_pointcloud(
                                                f"{mesh_name.rsplit('_', 1)[0]}.pt"
                                            )
                                            full_mesh_labels[f"{mesh_name.rsplit('_', 1)[0]}.pt"] = gt_pcd[:, -1]
                                        else:
                                            gt_pcd = self.valid_data_loader.dataset.get_gt_pointcloud(
                                                mesh_name)
                                            full_mesh_labels[mesh_name] = gt_pcd[:, -1]

                                    level_0_coords = self.valid_data_loader.dataset.get_level_0(
                                        mesh_name)

                                    if self._s3dis_gt_pcd:
                                        ball_tree = BallTree(
                                            level_0_coords[:, :3])
                                        dist, ind = ball_tree.query(
                                            gt_pcd[:, :3], k=1)

                                        torch_ind = torch.from_numpy(
                                            ind.flatten())
                                        torch_dist = torch.from_numpy(
                                            dist.flatten())

                                        class_predictions_block = class_predictions[torch_ind]
                                        batch_block = sample.batch[torch_ind]

                                        if mesh_name not in [
                                            'Area_5_hallway_1_0.pt',
                                            'Area_5_hallway_1_1.pt',
                                            'Area_5_hallway_2_0.pt',
                                                'Area_5_hallway_2_1.pt']:
                                            full_predictions[mesh_name] = torch.zeros((gt_pcd.shape[0],
                                                                                       len(
                                                                                           self.valid_data_loader.dataset.classes)),
                                                                                      dtype=torch.uint8)
                                        else:
                                            if f"{mesh_name.rsplit('_', 1)[0]}.pt" not in full_predictions:
                                                full_predictions[f"{mesh_name.rsplit('_', 1)[0]}.pt"] = torch.zeros(
                                                    (gt_pcd.shape[0],
                                                     len(self.valid_data_loader.dataset.classes)),
                                                    dtype=torch.uint8)
                                            full_predictions[f"{mesh_name.rsplit('_', 1)[0]}.pt"][
                                                torch_dist.flatten() < 0.12,
                                                class_predictions_block[torch_dist.flatten() < 0.12]] += 1
                            else:
                                # SCANNET BENCHMARK
                                m = mesh_name.split('.')[0]
                                full_predictions[mesh_name] = torch.zeros((
                                    self.valid_data_loader.dataset.get_full_mesh_size(
                                        m),
                                    len(self.valid_data_loader.dataset.classes)), dtype=torch.long)

                        if self._is_scannet:
                            full_predictions[mesh_name][
                                torch.arange(
                                    0, class_predictions_block[batch_block == i].shape[0], dtype=torch.long),
                                class_predictions_block[batch_block == i]] += 1
                        else:
                            if mesh_name not in [
                                'Area_5_hallway_1_0.pt',
                                'Area_5_hallway_1_1.pt',
                                'Area_5_hallway_2_0.pt',
                                    'Area_5_hallway_2_1.pt']:
                                full_predictions[mesh_name][
                                    torch.arange(
                                        0, class_predictions_block[batch_block == i].shape[0], dtype=torch.long),
                                    class_predictions_block[batch_block == i]] += 1

                        if not self._benchmark:
                            print(f"Current mesh: {mesh_name}")

                            argmax_predictions = torch.argmax(full_predictions[mesh_name], dim=1)
                            visualizer.visualize_result(mesh_name,
                                                        argmax_predictions,
                                                        self.valid_data_loader.dataset.get_full_mesh_label(mesh_name))
                        else:
                            self.valid_data_loader.dataset.visualize_result(mesh_name,
                                                                            full_predictions[mesh_name])

                        if not full_predictions[mesh_name].sum(dim=1).min().cpu().numpy() == round_num+1:
                            print('ERROR')
                            print(mesh_name)
                            _exit(self.writer, 1)

    def _runtime_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        runtimes = {}

        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(self.data_loader)):
                sample.to(self.device, inv=True)
                try:
                    start_time = time.time()
                    output = self.model(sample)
                    inference_time = time.time() - start_time
                    runtimes[sample.name[0].replace(
                        '_full', '')] = inference_time
                except RuntimeError as e:
                    print(str(e))
                    if 'out of memory' in str(e):
                        print('| EVAL OUT OF MEMORY')
                        _exit(self.writer, 1)
                    _exit(self.writer, 1)

        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(self.valid_data_loader)):
                sample.to(self.device, inv=True)
                try:
                    start_time = time.time()
                    _ = self.model(sample)
                    inference_time = time.time() - start_time
                    runtimes[sample.name[0].replace(
                        '_full', '')] = inference_time
                except RuntimeError as e:
                    print(str(e))
                    if 'out of memory' in str(e):
                        print('| EVAL OUT OF MEMORY')
                        _exit(self.writer, 1)
                    _exit(self.writer, 1)

        return runtimes
