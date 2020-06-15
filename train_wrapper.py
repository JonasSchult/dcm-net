"""As we do not operate on fix-sized meshes, our training crops may not fit on the GPU.
In that case, we track the maximal number of vertices which fit on the GPU
and restart the training process after the crash.
"""

import os
import argparse
import json
import glob
from stat import S_ISREG, ST_MODE, ST_MTIME


def main():
    parser = argparse.ArgumentParser(
        description='Wrapper for training process to keep track of maximal number of vertices')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-z', '--only_training',
                        dest='only_training', action='store_true')
    parser.set_defaults(only_training=False)
    parser.add_argument('-s', '--s3dis_gt_pcd',
                        dest='s3dis_gt_pcd', action='store_true')
    parser.set_defaults(s3dis_gt_pcd=False)

    args = parser.parse_args()

    max_num_points_per_batch = ""
    resume_arg = ""
    config = json.load(open(args.config))
    path = os.path.join(config['trainer']['save_dir'],
                        config['name'], str(config['id']))

    if os.path.exists(path):
        checkpoint_curr_path = f"{path}/checkpoint-curr.pth"
        if os.path.isfile(checkpoint_curr_path):
            resume_arg = f"-r {checkpoint_curr_path}"
        else:
            resume_paths = sorted([x for x in glob.glob(f"{path}/*.pth")])
            resume_paths = ((os.stat(path), path) for path in resume_paths)

            # leave only regular files, insert creation date
            entries = ((stat[ST_MTIME], path)
                    for stat, path in resume_paths if S_ISREG(stat[ST_MODE]))

            resume_path = sorted(entries)[-1][1]
            resume_arg = f"-r {resume_path}"

        max_points_file_path = f"{path}/max_num_points.txt"
        if os.path.isfile(max_points_file_path):
            with open(max_points_file_path, 'r') as points_file:
                if len(points_file.readline()) == 0:
                    os.remove(max_points_file_path)
                else:
                    point_epoch, max_points = [
                        int(x) for x in points_file.readline().split(';')]
                    max_num_points_per_batch = f"-p {max_points} -o {point_epoch}"
                    print(f"max num points: {max_num_points_per_batch}")

    print(f"Resume from {resume_arg} ...")

    error_code = os.system(
        f"python run.py -c {args.config} {resume_arg} {max_num_points_per_batch} {'-s' if args.s3dis_gt_pcd else ''}")
    error_code = error_code >> 8

    # COMMUNICATION WITH TRAINING PROCESS IS DONE VIA ERROR CODES
    while True:
        if error_code == 10:
            print('error code 10 received')
            # eval finished -> next training epoch
            resume_paths = sorted(
                [x for x in glob.glob(f"{path}/*.pth") if 'epoch' in x])
            max_epoch = -1
            winner = ''
            for resume_path in resume_paths:
                epoch = int(resume_path.split('epoch')[-1].split('.')[0])
                if epoch > max_epoch:
                    max_epoch = epoch
                    winner = resume_path

            max_num_points_per_batch = ""
            max_points_file_path = f"{path}/max_num_points.txt"
            if os.path.isfile(max_points_file_path):
                with open(max_points_file_path, 'r') as points_file:
                    point_epoch, max_points = [
                        int(x) for x in points_file.readline().split(';')]
                    max_num_points_per_batch = f"-p {max_points} -o {point_epoch}"
                    print(f"max num points: {max_num_points_per_batch}")

            error_code = os.system(
                f"python run.py -c {args.config} -r {winner} {max_num_points_per_batch} {'-s' if args.s3dis_gt_pcd else ''}")
            error_code = error_code >> 8
        elif error_code == 91:
            print('error code 91 received')
            # CUDA MEMORY batch size problem -> simply restart training process and hope for the best :)

            max_num_points_per_batch = ""
            max_points_file_path = f"{path}/max_num_points.txt"
            if os.path.isfile(max_points_file_path):
                with open(max_points_file_path, 'r') as points_file:
                    point_epoch, max_points = [
                        int(x) for x in points_file.readline().split(';')]
                    max_num_points_per_batch = f"-p {max_points} -o {point_epoch}"
                    print(f"max num points: {max_num_points_per_batch}")

            error_code = os.system(
                f"python run.py -c {args.config} -r {path}/checkpoint-curr.pth {max_num_points_per_batch} {'-s' if args.s3dis_gt_pcd else ''}")
            error_code = error_code >> 8
        elif error_code == 92:
            if args.only_training:
                error_code = 10
            else:
                print('error code 92 received')
                # training an epoch was successful -> do evaluation
                resume_paths = sorted(
                    [x for x in glob.glob(f"{path}/*.pth") if 'epoch' in x])
                max_epoch = -1
                winner = ''
                for resume_path in resume_paths:
                    epoch = int(resume_path.split('epoch')[-1].split('.')[0])
                    if epoch > max_epoch:
                        max_epoch = epoch
                        winner = resume_path

                error_code = os.system(
                    f"python run.py -c {args.config} -r {winner} -e {'-s' if args.s3dis_gt_pcd else ''}")
                error_code = error_code >> 8
        else:
            print('some other error code received')
            # some mysterious other problem occured -> exit :(
            exit(1)


if __name__ == '__main__':
    main()
