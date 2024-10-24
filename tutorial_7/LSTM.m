function wLSTM = LSTM(nn_input)
    % Set your python path here where all necessary dependencies are 
    % installed:
    pyenv('Version', "PYTHON PATH");
    
    % Clip the angles the NN does not support
    nn_input = max(min(nn_input,150),30);
    
    currentFolder = pwd;
    res = pyrunfile("run_lstm.py","output",x=nn_input);
    
    result = cellfun(@double,cell(res)); %pylist-> cell array -> double array
    wLSTM = complex(result(1:16),result(17:32));
    wLSTM = reshape(wLSTM, [16,1]);
end
