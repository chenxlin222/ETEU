# ETEU-tracker

Matlab implementation of our ETEU tracker.

The results, precision and success plots for six aerial tracking benchmarks have been released. And the code is coming soon.

## Demonstration running instructions

This code is compatible with six aerial tracking benchmarks, i.e., UAV123, UAV123@10fps, UAVDT, DTB70, VisDrone2018-SOT-test, and UAVTrack112. It is easy to run this tracker as long as you have MATLAB. Just download the package and do the following steps:

  1. Choose the seq name in configSeqs_demo_for_ETEU.m
  2. Run ETEU_Demo_single_seq.m

## Results on UAV datasets

### UAV123

The tracking experiment in the environment with multiple static similar people.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/multi_person_tracking.gif)

The tracking experiment in the environment with a single moving person.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/single_moving_person_tracking.gif)

The tracking experiment in the environment with multiple moving similar people.
![img](https://github.com/chenxlin222/ETEU-tracker/blob/main/img/multi_person_tracking_moving.gif)
