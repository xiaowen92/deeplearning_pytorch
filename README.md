# deeplearning_pytorch
using PyTorch to learn Deep learning


## Useful command for VSCODE Python environment for Mac 
1. python3 --version  -> check python version in mac, please do use python3 
2. python3.11 --version -> check specific python version in your mac 
3. python3.11 -m venv .venv_tf -> create a python virtual env in your macbook for a specific Python version. here the virtual environment named ".venv_tf"
4. pip3 install <package> when you install package, please use pip3. Mac distinguished python2 and 3 with pip3

## Usful information for Machine learning 
1. Feature Engineering 
    Sometimes instead of letting machine learn itself, you could help to label it yourself, to let the machine learn more faster. for example, rush hour is 6-8pm and 7-9am, you can manually create a new col to indicate whether the specific time is rush hr, instead the machine learn itself. 


## Useful info for pytorch beginners
1. x = torch.tensor([1, 2, 3]) -> .tensor can convert list, it can also convert ** Numpy array **
2. x = torch.from_numpy(numpy_array), numpy array 
3. for dataframe, need to use df.values -> convert df to np array and then use tensor() to convert to tensor
torch.squeeze(dim=0) , 
4. * **Dot Product (`torch.matmul()`)**: Calculates the dot product of two vectors or matrices.
5. boolean logical operators "|, &, =, ..." 
6. stastic operation tensor_obj.mean(), .std()
7. datatype, tensor_obj.dtype

