clear all;
clc;

%% Load Data

pred = csvread('../Predict/ConvLSTM_mm.csv');
test = csvread('../Predict/ConvLSTM_mm_Truth.csv');

%% Generate Gif

filename = 'ConvNTM_mm.gif';
for idx=1:20
    if idx==1
        imwrite(reshape(uint8(pred(idx, :)*255), [64, 64]), filename, 'gif', 'Loopcount', inf);
    else
        imwrite(reshape(uint8(pred(idx, :)*255), [64, 64]), filename, 'gif', 'Writemode', 'append');
    end
end

filename = 'ConvNTM_mm_Truth.gif';
for idx=1:20
    if idx==1
        imwrite(reshape(uint8(test(idx, :)*255), [64, 64]), filename, 'gif', 'Loopcount', inf);
    else
        imwrite(reshape(uint8(test(idx, :)*255), [64, 64]), filename, 'gif', 'Writemode', 'append');
    end
end

