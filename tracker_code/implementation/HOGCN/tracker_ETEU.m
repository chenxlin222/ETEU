function results = tracker_ETDTT(params)

global sum_psr2;
%global sum_apce;
global sum_Fmax;
global counts;
%% Initialization
% Get sequence info
learning_rate = params.learning_rate;


[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end
% Init position
pos = seq.init_pos(:)';
% context position
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

temporal_regularization_factor = params.temporal_regularization_factor; 

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');


% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

small_filter_sz{1} = floor(base_target_sz/(feature_cell_sz(1,1)));

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

filter_sz_cell_ours{1} = filter_sz_cell{1}; 

% initialize previous response map
% M_prev= zeros(filter_sz_cell{1});

% The size of the label function DFT. Equal to the maximum filter size
[output_sz_hand, k1] = max(filter_sz, [], 1);

output_sz = output_sz_hand;

k1 = k1(1);
% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

%%%模板更新相关参数初始化
curr_Fmax_all = zeros(seq.num_frames, 1); %初始化每帧的Fmax矩阵集合
curr_psr2_all = zeros(seq.num_frames, 1); %初始化每帧的psr2矩阵集合，经查找资料，psr在20-60之间时跟踪效果比较好，小于10时，说明目标跟踪失败了或目标被遮挡了
mean_psr2_all = zeros(seq.num_frames, 1); %初始化psr2平均值矩阵集合
mean_Fmax_all = zeros(seq.num_frames, 1); %初始化Fmax平均值矩阵集合
Deta_res_all = zeros(seq.num_frames, 1); %初始化每帧特征所计算得到的标准差矩阵，该值用来计算当前帧和前一帧特征协方差的相关系数
p_res_respre = ones(seq.num_frames, 1); %初始化两帧之间的相关系数
cf_f_all = zeros(feature_sz(1), feature_sz(2), seq.num_frames);
cf_f_mean_all = zeros(seq.num_frames, 1);
% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf          = fft2(y); 
end

% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);
% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
% lfl: parameters for scale estimation
scale_sigma = sqrt(params.num_scales) * params.scale_sigma_factor;
ss = (1:params.num_scales) - ceil(params.num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(params.num_scales,2) == 0
    scale_window = single(hann(params.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(params.num_scales));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%

ss = 1:params.num_scales;
scaleFactors = params.scale_step.^(ceil(params.num_scales/2) - ss);
if params.scale_model_factor^2 * prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end

if prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end
scale_model_sz = floor(params.init_sz * params.scale_model_factor);
% scale_model_sz

% set maximum and minimum scales
min_scale_factor = params.scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(params.scale_step));
max_scale_factor = params.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(params.scale_step));

seq.time = 0;

% Define the learning variables
% h_current_key = cell(num_feature_blocks, 1);
cf_f = cell(num_feature_blocks, 1);
cf_f_first = cell(num_feature_blocks, 1);%%%%%增加了第一帧的cf_f
% Allocate
%scores_fs_feat = cell(1,1,num_feature_blocks);
scores_fs_feat = cell(1,1,3);
%% Main loop here
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    %%%%%%%%%%变量值
%     imim = im;
    a1 = 0.82;
    
    tic();
    
    % Target localization step
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
%             sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            responsef_padded = resizeDFT2(scores_fs, output_sz);
            response = ifft2(responsef_padded, 'symmetric');
                      
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%对滤波器的操作 
            Deta_res_all(seq.frame) = sqrt(sum(sum((cf_f_all(:, :, seq.frame-1) - cf_f_mean_all(seq.frame-1)) .^ 2)) / (feature_sz(1) * feature_sz(2)));
            %%%%%%%%%%%%%%%%%%%%
            if seq.frame > 11
                cov_res_respre = (sum(sum((cf_f_all(:, :, seq.frame-1) - cf_f_mean_all(seq.frame-1)) .* (cf_f_all(:, :, seq.frame-10) - cf_f_mean_all(seq.frame-10)))))/(feature_sz(1)*feature_sz(2));
                cov_res_respre1 = (sum(sum((cf_f_all(:, :, seq.frame-1) - cf_f_mean_all(seq.frame-1)) .* (cf_f_all(:, :, 1) - cf_f_mean_all(1)))))/(feature_sz(1)*feature_sz(2));
                p_res_respre(seq.frame) = a1 * cov_res_respre / (Deta_res_all(seq.frame) .* Deta_res_all(seq.frame-9)) + (1-a1) * cov_res_respre1 / (Deta_res_all(seq.frame) .* Deta_res_all(2));
            elseif seq.frame < 3
                p_res_respre(seq.frame) = 1;
            else
                cov_res_respre = (sum(sum((cf_f_all(:, :, seq.frame-1) - cf_f_mean_all(seq.frame-1)) .* (cf_f_all(:, :, seq.frame-2) - cf_f_mean_all(seq.frame-2)))))/(feature_sz(1)*feature_sz(2));
                p_res_respre(seq.frame) = cov_res_respre / (Deta_res_all(seq.frame) .* Deta_res_all(seq.frame-1));
            end

            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);

            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor;            
%             scale_change_factor = scaleFactors(sind);
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % lfl: SCALE SPACE SEARCH
            xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
            xsf = fft(xs,[],2);
%             size(xsf)
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + params.scale_lambda) ));            
            % find the maximum scale response
            recovered_scale = find(scale_response == max(scale_response(:)), 1);
%             scale_response
%             recovered_scale
            % update the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end            

            iter = iter + 1;
        end
        
    end

    %% Model update step
    % extract image region for training sample
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    
    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    % train the CF model for each feature
    for k = 1: 1
        if (seq.frame == 1)
            model_xf = xlf{k};
            f_pre_f = single(zeros(size(model_xf)));
            gamma = 0;
            
        %%%%%%%%%%%%%%%修改更新策略
        else
            [curr_psr2, curr_Fmax] = psr2(response);
            %[curr_apce, ~] = apce(response);
            %%%%%%%apce、Fmax这个相关语句是用来做对比的
            %curr_psr2
            sum_psr2 = sum_psr2 + curr_psr2;
            sum_Fmax = sum_Fmax + curr_Fmax;
            %sum_apce = sum_apce + curr_apce;
            mean_psr2 = sum_psr2 / counts;
            mean_Fmax = sum_Fmax / counts;
            %mean_apce = sum_apce / counts;
            counts = counts + 1;
            curr_Fmax_all(seq.frame) = curr_Fmax;
            curr_psr2_all(seq.frame) = curr_psr2;
            %curr_apce_all(seq.frame) = curr_apce;
     
            beta1 = 0.5;
            beta2 = 0.6; %%%beta1和beta2暂时没有用到
            
            mean_Fmax_all(seq.frame) = beta2 * mean_Fmax;
            mean_psr2_all(seq.frame) = beta1 * mean_psr2;
            %mean_apce_all(seq.frame) = beta2 * mean_apce;
            gamma = temporal_regularization_factor(k);
            
%             %%%%%%%%%%%%不使用其他判断条件，直接更新模板的语句
%             model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xlf{k});

            %%%%%%%%%%%%%%%%%%%%%%%%%%使用的更新条件
            if (curr_psr2 >= 35.5)
                model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xlf{k});
            elseif ((p_res_respre(seq.frame) >= 0.96) && (p_res_respre(seq.frame) <= 0.98))
                model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xlf{k});
            else
                model_xf = model_xf;
            end
            
        end
        
        g_f = single(zeros(size(xlf{k})));
        f_f = g_f;
        l_f = g_f;
        mu    = 1;
        betha = 10;
        mumax = 10000;
        i = 1;
        
        N = prod(filter_sz_cell_ours{k});
        S_xx = sum(conj(model_xf) .* model_xf, 3);
        % ADMM solving process
        while (i <= params.admm_iterations)
            
            B = S_xx + N * mu;
            S_lx = sum(conj(model_xf) .* l_f, 3);
            S_hx = sum(conj(model_xf) .* f_f, 3);
            g_f = (((1/(N*mu)) * bsxfun(@times, yf, model_xf)) - ((1/mu) * l_f) + f_f) - ...
                bsxfun(@rdivide,(((1/(N*mu)) * bsxfun(@times, model_xf, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf, S_lx)) + (bsxfun(@times, model_xf, S_hx))), B);
            
            %   solve for H
            h = (N/((mu*N)+ params.admm_lambda + gamma/sqrt(N)))* ifft2((mu*g_f) + l_f + (gamma/sqrt(N))*f_pre_f);
            [sx,sy,h] = get_subwindow_no_window(h, floor(filter_sz_cell_ours{k}/2) , small_filter_sz{k});
            t = single(zeros(filter_sz_cell_ours{k}(1), filter_sz_cell_ours{k}(2), size(h,3)));
            t(sx,sy,:) = h;
            f_f = fft2(t);
            
            %   update L
            l_f = l_f + (mu * (g_f - f_f));
            cf_f{k} = g_f;
            cf_f_all(:, :, seq.frame) = 10 * sum(real(g_f),3);
            cf_f_mean_all(seq.frame) = mean(mean(cf_f_all(:, :, seq.frame)));
            if seq.frame == 1
                cf_f_first{k} = cf_f{k};
            end
            f_pre_f = f_f;
            %   update mu- betha = 10.
            mu = min(betha * mu, mumax);
            i = i+1;
        end
    end
 
    %% Upadate Scale
    xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    if seq.frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
    end
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %fprintf('target_sz: %f, %f \n', target_sz(1), target_sz(2));
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%添加--画图
%      figure(3)
%     rect_box = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
%     subplot(221);
%     imim = insertShape(imim, 'Rectangle', rect_box, 'LineWidth', 3, 'Color', 'red');
%     imshow(imim);
%     x = ['frame', num2str(seq.frame)];
%     title(x);
%     hold on;
%     subplot(223);
%     for k =1:1
%         if seq.frame > 1
%             mesh(fftshift(response_first));
%             grid off;
%             axis off;
%             x = ['response of frame',num2str(seq.frame)];
%             title(x);
%         end
%     end
    %subplot(222);
%     plot(curr_psr2_all, 'k-', 'LineWidth',1);
%     legend('PSR^2');
%     plot(100./mean_psr2_all, 'k-', 'LineWidth',1);
%     legend('PSR^2'); 
    %subplot(224);
%     plot(p_res_respre, 'k-', 'LineWidth',1);
%     legend('Cov(x, y)')
%     saveas(gcf,['./seq/response/wakeboard_psr_cov_1' num2str(seq.frame) '.jpg'] );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        
        imagesc(im_to_show);
        hold on;
        rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
        text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
                    
        drawnow
    end
end
%p_res_respre
%%%%%单独画三个PSR2，Fmax，apce的值的变化情况
%subplot(221);
% plot(curr_psr2_all, 'k-', 'LineWidth',1);
% hold on;
% plot(5000./mean_psr2_all, 'r-', 'LineWidth',1);
% legend('PSR^2','PSR^2\_Threshold');
% xlabel('Frame');
% ylabel('PSR^2');
% x = ['PSR^2 and PSR^2\_Threshold (with updated)'];
% title(x)
%curr_psr2_all

%cf_f_all
% aaa = (sum(sum((cf_f_all(:, :, 2) - cf_f_mean_all(2)) .* (cf_f_all(:, :, 1) - cf_f_mean_all(1)))))/(feature_sz(1)*feature_sz(2))
% bbb = sqrt(sum(sum((cf_f_all(:, :, 1) - cf_f_mean_all(1)) .^ 2)) / (feature_sz(1) * feature_sz(2)))
% bb = sqrt(sum(sum((cf_f_all(:, :, 2) - cf_f_mean_all(2)) .^ 2)) / (feature_sz(1) * feature_sz(2)))
% bbbb = Deta_res_all(2)
% bbbbb = Deta_res_all(3)
% ccc = aaa / (Deta_res_all(2) .* Deta_res_all(3))
% p_res_respre

%aaa = real((sum(sum((cf_f_all(:, :, 2) - cf_f_mean_all(2)) .* (cf_f_all(:, :, 1) - cf_f_mean_all(1)))))/(feature_sz(1)*feature_sz(1)))

% hold on;
% subplot(222);
% plot(curr_Fmax_all, 'b-', 'LineWidth',1);
% hold on;
% plot(mean_Fmax_all, 'r-', 'LineWidth',1);
% legend('Fmax','Fmax\_Threshold');
% subplot(223);
% plot(curr_apce_all, 'b-', 'LineWidth',1);
% hold on;
% plot(mean_apce_all, 'r-', 'LineWidth',1);
% legend('APCE','APCE\_Threshold');

[~, results] = get_sequence_results(seq);

%disp(['fps: ' num2str(results.fps)])

