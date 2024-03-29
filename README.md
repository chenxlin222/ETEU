# ETEU-tracker

Matlab implementation of our ETEU tracker.

The results, precision and success plots for six aerial tracking benchmarks have been released. 

## 1. Demonstration running instructions

This code is compatible with six aerial tracking benchmarks, i.e., UAV123, UAV123@10fps, UAVDT, DTB70, VisDrone2018-SOT-test, and UAVTrack112. It is easy to run this tracker as long as you have MATLAB. Just download the package and do the following steps:

  1. Choose the seq name in configSeqs_demo_for_ETEU.m
  2. Run ETEU_Demo_single_seq.m

## 2. Results on UAV datasets

### (1) UAV123

<img src="https://github.com/chenxlin222/ETEU/blob/main/figs/UAV123/precision_OPE.png" width="375px"> <img src="https://github.com/chenxlin222/ETEU/blob/main/figs/UAV123/success_OPE.png" width="375px">
    
### (2) UAVDT

<img src="https://github.com/chenxlin222/ETEU/blob/main/figs/UAVDT/quality_plot_error_OPE.png" width="375px"> <img src="https://github.com/chenxlin222/ETEU/blob/main/figs/UAVDT/quality_plot_overlap_OPE.png" width="375px">
    
### (3) UAVTrack112

<img src="https://github.com/chenxlin222/ETEU/blob/main/figs/UAVTrack112/quality_plot_error_OPE.png" width="375px"> <img src="https://github.com/chenxlin222/ETEU/blob/main/figs/UAVTrack112/quality_plot_overlap_OPE.png" width="375px">

### (4) VisDrone2018-SOT-test

<img src="https://github.com/chenxlin222/ETEU/blob/main/figs/VisDrone2018-SOT-test/quality_plot_error_OPE_threshold.png" width="375px"> <img src="https://github.com/chenxlin222/ETEU/blob/main/figs/VisDrone2018-SOT-test/quality_plot_overlap_OPE_AUC.png" width="375px">

### (5) DTB70

<img src="https://github.com/chenxlin222/ETEU/blob/main/figs/DTB70/quality_plot_error_OPE_threshold.png" width="375px"> <img src="https://github.com/chenxlin222/ETEU/blob/main/figs/DTB70/quality_plot_overlap_OPE_AUC.png" width="375px">

## 3. The tracking mission in three different environments
The tracking experiment in the environment with multiple static similar people.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/multi_person_tracking.gif)

The tracking experiment in the environment with a single moving person.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/single_moving_person_tracking.gif)

The tracking experiment in the environment with multiple moving similar people.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/multi_person_tracking_moving.gif)

## 4. The tracking mission in real complex environments
scenario one (Drone view 2.5X):

![img](https://github.com/chenxlin222/ETEU/blob/main/img/scenario_one_2point5X.gif)

scenario two (Drone view 2X):

![img](https://github.com/chenxlin222/ETEU/blob/main/img/scenario_two_2X.gif)

global view (2.5X):

![img](https://github.com/chenxlin222/ETEU/blob/main/img/scenario_one_2point5X_global.gif)
