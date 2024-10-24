function wFFNN = FFNN(nn_input)
    % Set your python path here where all necessary dependencies are 
    % installed:
    pyenv('Version', "PYTHON_PATH");
    
    % Clip the angles the NN does not support
    nn_input = max(min(nn_input,150),30);
    
    currentFolder = pwd;
    res = pyrunfile(currentFolder+"/run_ffnn.py","output",x=nn_input);
    
    result = cellfun(@double,cell(res)); %pylist-> cell array -> double array
    wFFNN = complex(result(1:16),result(17:32));
    wFFNN = reshape(wFFNN, [16,1]);
end
