
function results = run_ETEU(seq, rp, sname, bSaveImage)

global sum_psr2; sum_psr2 = 0;
global sum_Fmax; sum_Fmax = 0;
global counts; counts = 1;
%global sum_apce; sum_apce = 0;

% Feature specific parameters
hog_params.cell_size = 4;
hog_params.compressed_dim = 10;
hog_params.nDim = 31;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.nDim = 10;

% Which features to include
params.t_features = {
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...    
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
};

% Global feature parameters1s
params.t_global.cell_size = 4;                  % Feature cell size

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 5;         % The scaling of the target size to get the search area
params.min_image_sample_size = 150^2;   % Minimum area of image samples
params.max_image_sample_size = 200^2;   % Maximum area of image samples

% Spatial regularization window_parameters
params.feature_downsample_ratio = [4]; %  Feature downsample ratio
params.reg_window_max = 1e5;           % The maximum value of the regularization window
params.reg_window_min = 1e-3;           % the minimum value of the regularization window

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = 1/15;		% Label function sigma
params.temporal_regularization_factor = [28 28]; % The temporal regularization parameters

% ADMM parameters
params.max_iterations = [2 2];
params.init_penalty_factor = [1 1];
params.max_penalty_factor = [0.1, 0.1];
params.penalty_scale_step = [10, 10];

params.num_scales = 33;
params.hog_scale_cell_size = 4;
params.learning_rate_scale = 0.025;
params.scale_sigma_factor = 1/2;
params.scale_model_factor = 1.0;
params.scale_step = 1.03;
params.scale_model_max_area = 32*16;
params.scale_lambda = 1e-4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%sname
% if strcmp(sname,'bike1') | strcmp(sname,'car6_2') | strcmp(sname,'car6_4') | strcmp(sname,'group1_3') | strcmp(sname,'uav2') | strcmp(sname,'person3_s')
%     params.learning_rate = 0.011;
% elseif strcmp(sname,'boat8') | strcmp(sname,'person13') | strcmp(sname,'person17_1') | strcmp(sname,'person7_2')
%     params.learning_rate = 0.025;
% else
%     params.learning_rate = 0.0192;
% end
params.learning_rate = 0.0192;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.admm_iterations = 2;
params.admm_lambda = 0.01;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
params.visualization = 0;               % Visualiza tracking and detection scores

% GPU
params.use_gpu = false;                 % Enable GPU or not
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default

% Initialize
params.seq = seq;

% Run tracker
results = tracker_ETEU(params);
