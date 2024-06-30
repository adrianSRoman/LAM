import csv
import os
import json5
import argparse
import numpy as np
from scipy import stats
from IPython import embed

import SELD_evaluation_metrics


# -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        if len(_words) == 6: # frame, class idx, source_id, polar coordinates(2), distance
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        elif len(_words) == 7: # frame, class idx, source_id, cartesian coordinates(3), distance
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict

def segment_labels(_pred_dict, _max_frames):
    '''
    Collects class-wise sound event location information in segments of length 1s from reference dataset
    :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
    :param _max_frames: Total number of frames in the recording
    :return: Dictionary containing class-wise sound event location information in each segment of audio
    dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
    '''
    nb_label_frames_1s = 10 # 10 frames per second
    nb_blocks = int(np.ceil(_max_frames/float(nb_label_frames_1s)))
    output_dict = {x: {} for x in range(nb_blocks)}
    for frame_cnt in range(0, _max_frames, nb_label_frames_1s):
        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        block_cnt = frame_cnt // nb_label_frames_1s
        loc_dict = {}
        for audio_frame in range(frame_cnt, frame_cnt+nb_label_frames_1s):
            if audio_frame not in _pred_dict:
                continue
            for value in _pred_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}

                block_frame = audio_frame - frame_cnt
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for class_cnt in loc_dict:
            if class_cnt not in output_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []

            keys = [k for k in loc_dict[class_cnt]]
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

            output_dict[block_cnt][class_cnt].append([keys, values])

    return output_dict


def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:

                ele_rad = tmp_val[3]*np.pi/180.
                azi_rad = tmp_val[2]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
    return out_dict


def convert_output_format_cartesian_to_polar(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                # in degrees
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                r = np.sqrt(x**2 + y**2 + z**2)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], azimuth, elevation])
    return out_dict


#------------------------------------------------------------------------------------------------

def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


class ComputeSELDResults(object):
    def __init__(
            self, params, ref_files_folder=None, use_polar_format=True
    ):
        self._use_polar_format = use_polar_format
        self._desc_dir = os.path.join(params['ground_truths'], 'metadata_dev')
        self._doa_thresh = params['lad_doa_thresh']
        self._num_classes = params['num_classes']

        # Load feature class
        #self._feat_cls = cls_feature_class.FeatureClass(params)
        
        # collect reference files
        self._ref_labels = {}
        for split in os.listdir(self._desc_dir):      
            for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                # Load reference description file
                gt_dict = load_output_format_file(os.path.join(self._desc_dir, split, ref_file))
                if not self._use_polar_format:
                    gt_dict = convert_output_format_polar_to_cartesian(gt_dict)
                nb_ref_frames = max(list(gt_dict.keys()))
                self._ref_labels[ref_file] = [segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        pred_labels_dict = {}
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._num_classes, doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = load_output_format_file(os.path.join(pred_files_path, pred_file))
            if self._use_polar_format:
                pred_dict = convert_output_format_cartesian_to_polar(pred_dict)
            pred_labels = segment_labels(pred_dict, self._ref_labels[pred_file][1])
            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels
        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
            global_values = [ER, F, LE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_files:
                leave_one_out_list = pred_files[:]
                leave_one_out_list.remove(leave_file)
                eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._num_classes, doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0])
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                leave_one_out_est = [ER, F, LE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)
                    
            estimate, bias, std_err, conf_interval = [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                           global_value=global_values[i],
                           partial_estimates=partial_estimates[:, i],
                           significance_level=0.05
                           )
            return [ER, conf_interval[0]], [F, conf_interval[1]], [LE, conf_interval[2]], [LR, conf_interval[3]], [seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5,13,2) if len(classwise_results) else []]
      
        else:      
            return ER, F, LE, LR, seld_scr, classwise_results


    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
            Get all categories of results.
            ;score_type_list: Supported
                'all' - all the predicted files
                'room' - for individual rooms

        '''

        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        nb_pred_files = len(pred_files)

        # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)

        print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, self._nb_ref_files))
        print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

        for score_type in score_type_list:
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type=='all' else 'score per {}'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            split_cnt_dict = self.get_nb_files(pred_files, tag=score_type) # collect files corresponding to score_type
            # Calculate scores across files for a given score_type
            for split_key in np.sort(list(split_cnt_dict)):
                # Load evaluation metric class
                eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._num_classes, doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    # Load predicted output format file
                    pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_output_format_files, pred_file))
                    if self._use_polar_format:
                        pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
                    pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])

                    # Calculated scores
                    eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])

                # Overall SED and DOA scores
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

                print('\nAverage score for {} {} data using {} coordinates'.format(score_type, 'fold' if score_type=='all' else split_key, 'Polar' if self._use_polar_format else 'Cartesian' ))
                print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
                print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(ER, 100*F))
                print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoA metrics based on DCASE task3")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    args = parser.parse_args()
    config = json5.load(open(args.configuration))

    pred_output_format_files = config["predictions"]  # Path of the DCASEoutput format files
    score_obj = ComputeSELDResults(config) # Compute just the DCASE final results

    use_jackknife=False
    ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(pred_output_format_files,is_jackknife=use_jackknife )
   
    print('SELD score (early stopping metric): {:0.2f} {}'.format(seld_scr[0] if use_jackknife else seld_scr, '[{:0.2f}, {:0.2f}]'.format(seld_scr[1][0], seld_scr[1][1]) if use_jackknife else ''))
    print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(ER[0]  if use_jackknife else ER, '[{:0.2f},  {:0.2f}]'.format(ER[1][0], ER[1][1]) if use_jackknife else '', 100*F[0]  if use_jackknife else 100*F, '[{:0.2f}, {:0.2f}]'.format(100*F[1][0], 100*F[1][1]) if use_jackknife else ''))
    print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(LE[0] if use_jackknife else LE, '[{:0.2f}, {:0.2f}]'.format(LE[1][0], LE[1][1]) if use_jackknife else '', 100*LR[0]  if use_jackknife else 100*LR,'[{:0.2f}, {:0.2f}]'.format(100*LR[1][0], 100*LR[1][1]) if use_jackknife else ''))
    if config['average']=='macro':
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(config['unique_classes']):
            print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
cls_cnt, 
classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '', 
classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '', 
classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '', 
classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '', 
classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))


    # UNCOMMENT to Compute DCASE results along with room-wise performance
    # score_obj.get_consolidated_SELD_results(pred_output_format_files)


