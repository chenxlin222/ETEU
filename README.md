# ETEU-tracker

Matlab implementation of our ETEU tracker.

The results, precision and success plots for six aerial tracking benchmarks have been released. And the code is coming soon.

## 1. Demonstration running instructions

This code is compatible with six aerial tracking benchmarks, i.e., UAV123, UAV123@10fps, UAVDT, DTB70, VisDrone2018-SOT-test, and UAVTrack112. It is easy to run this tracker as long as you have MATLAB. Just download the package and do the following steps:

  1. Choose the seq name in configSeqs_demo_for_ETEU.m
  2. Run ETEU_Demo_single_seq.m

## 2. Results on UAV datasets

### (1) UAV123

<img src="https://github.com/chenxlin222/ETEU/blob/main/figs/UAV123/precision_OPE.png" width="375px">
<img src="https://github.com/chenxlin222/ETEU/blob/main/figs/UAV123/success_OPE.png" width="375px">
    
### (2) UAVDT
  ![img](https://github.com/chenxlin222/ETEU/blob/main/figs/UAVDT/quality_plot_error_OPE.png)
   ![img](https://github.com/chenxlin222/ETEU/blob/main/figs/UAVDT/quality_plot_overlap_OPE.png)
    
### (3) UAVTrack112
   ![img](https://github.com/chenxlin222/ETEU/blob/main/figs/UAVTrack112/quality_plot_error_OPE.png)
   ![img](https://github.com/chenxlin222/ETEU/blob/main/figs/UAVTrack112/quality_plot_overlap_OPE.png)

### (4) VisDrone2018-SOT-test
   ![img](https://github.com/chenxlin222/ETEU/blob/main/figs/VisDrone2018-SOT-test/quality_plot_error_OPE_threshold.png)
   ![img](https://github.com/chenxlin222/ETEU/blob/main/figs/VisDrone2018-SOT-test/quality_plot_overlap_OPE_AUC.png)

### (5) DTB70
   ![img](https://github.com/chenxlin222/ETEU/blob/main/figs/DTB70/quality_plot_error_OPE_threshold.png)
   ![img](https://github.com/chenxlin222/ETEU/blob/main/figs/DTB70/quality_plot_overlap_OPE_AUC.png)


The tracking experiment in the environment with multiple static similar people.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/multi_person_tracking.gif)

The tracking experiment in the environment with a single moving person.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/single_moving_person_tracking.gif)

The tracking experiment in the environment with multiple moving similar people.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/multi_person_tracking_moving.gif)
