# deeplearning_pytorch
using PyTorch to learn Deep learning


## Useful command for VSCODE Python environment for Mac 
1. python3 --version  -> check python version in mac, please do use python3 
2. python3.11 --version -> check specific python version in your mac 
3. python3.11 -m venv .venv_tf -> create a python virtual env in your macbook for a specific Python version. here the virtual environment named ".venv_tf"
4. pip3 install <package> when you install package, please use pip3. Mac distinguished python2 and 3 with pip3


## Useful info for pytorch beginners
1. x = torch.tensor([1, 2, 3]) -> .tensor can convert list, it can also convert ** Numpy array **
2. x = torch.from_numpy(numpy_array), numpy array 
3. for dataframe, need to use df.values -> convert df to np array and then use tensor() to convert to tensor
torch.squeeze(dim=0) , 
4. * **Dot Product (`torch.matmul()`)**: Calculates the dot product of two vectors or matrices.