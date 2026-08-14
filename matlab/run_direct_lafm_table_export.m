function run_direct_lafm_table_export(input_dir, output_dir, nanolocz_lib)
% Export direct LAFM localization tables for every aligned TIFF stack.

addpath(nanolocz_lib);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, '*.tiff'));
columns = {'x','y','z','prominence','frame','source_frame', ...
    'time','correlation','extra9','extra10','extra11','extra12'};
manifest = repmat(struct('source','','status','','rows',0,'frames',0, ...
    'shape',[],'table','','error',''), numel(files), 1);

for file_index = 1:numel(files)
    source = fullfile(files(file_index).folder, files(file_index).name);
    manifest(file_index).source = files(file_index).name;
    manifest(file_index).source_bytes = files(file_index).bytes;
    manifest(file_index).source_mtime = files(file_index).datenum;
    try
        [~, stem] = fileparts(files(file_index).name);
        table_name = [stem '.csv'];
        table_path = fullfile(output_dir, table_name);
        if exist(table_path, 'file')
            existing = readtable(table_path);
            existing_info = imfinfo(source);
            manifest(file_index).status = 'ok';
            manifest(file_index).rows = height(existing);
            manifest(file_index).frames = numel(existing_info);
            manifest(file_index).shape = [
                existing_info(1).Height, existing_info(1).Width, ...
                numel(existing_info)
            ];
            manifest(file_index).table = table_name;
            fprintf('[%d/%d] %s: resumed (%d rows)\n', ...
                file_index, numel(files), files(file_index).name, ...
                height(existing));
            continue
        end
        info = imfinfo(source);
        n_frames = numel(info);
        first = double(imread(source, 1, 'Info', info));
        stack = zeros(size(first, 1), size(first, 2), n_frames);
        stack(:,:,1) = first;
        for frame = 2:n_frames
            stack(:,:,frame) = double(imread(source, frame, 'Info', info));
        end

        locs = zeros(0, numel(columns));
        for frame = 1:n_frames
            peaks = Fast_peaks2D(stack(:,:,frame), 0, 1, 0);
            if ~isempty(peaks)
                table = zeros(size(peaks,1), numel(columns));
                table(:,1:4) = peaks;
                table(:,5:8) = frame;
                locs = [locs; table]; %#ok<AGROW>
            end
        end

        if ~isempty(locs)
            locs = localize(stack, locs, 'bicubic', 1);
            locs = locs(all(isfinite(locs(:,1:5)),2),:);
            xs = round(locs(:,1));
            ys = round(locs(:,2));
            frames = round(locs(:,5));
            inside = xs > 0 & xs < size(stack,2) & ...
                ys > 0 & ys < size(stack,1) & ...
                frames > 0 & frames <= n_frames;
            locs = locs(inside,:);
            xs = xs(inside);
            ys = ys(inside);
            frames = frames(inside);
            indices = sub2ind(size(stack), ys, xs, frames);
            locs(:,3) = stack(indices);
        end

        writetable(array2table(locs, 'VariableNames', columns), ...
            table_path);
        manifest(file_index).status = 'ok';
        manifest(file_index).rows = size(locs,1);
        manifest(file_index).frames = n_frames;
        manifest(file_index).shape = size(stack);
        manifest(file_index).table = table_name;
        fprintf('[%d/%d] %s: %d rows\n', file_index, numel(files), ...
            files(file_index).name, size(locs,1));
    catch err
        manifest(file_index).status = 'error';
        manifest(file_index).error = getReport(err, 'extended', ...
            'hyperlinks', 'off');
        fprintf(2, '[%d/%d] %s: ERROR %s\n', file_index, numel(files), ...
            files(file_index).name, err.message);
    end
end

parameters = struct('low_pass_sigma',0,'high_pass_sigma',0, ...
    'min_separation',1,'height_threshold',0, ...
    'prominence_threshold',0,'localization_method','bicubic', ...
    'pixperfeat',1);
payload = struct('pipeline_version',1,'parameters',parameters,'files',manifest);
fid = fopen(fullfile(output_dir, 'manifest.json'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(payload, PrettyPrint=true));
end
