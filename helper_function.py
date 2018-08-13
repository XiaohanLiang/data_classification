# This script contains helper functions to help debugging
import pickle
import numpy as np
picDir = "pickle_file/"

#
# @Fucntion: This function helps to save pickle such that
#            we can test the script step by step
#
# @Params: Data      -> data you would like to save as a pickle
#          file_name -> specify the file_name
#
def save_as_pickle(data,file_name):
    with open(picDir+file_name+".p",'wb') as file:
        pickle.dump(data,file)

#
# @Fucntion: This function was used to read pickle
#
# @Params: specify the file_name you would like to read
#
def read_from_pickle(file_name):
    with open(picDir+file_name+".p",'r') as file:
        data = pickle.load(file)
    return data

#
# @Function: Help you to check the detail of array
#

def print_stuff(name):
    print("----------------------------------------------------")
    length = len(name)
    print("This array has dimension "+str(length))
    if(length!=1):
        length_2 = len(name[0])
        print("Where each column has "+str(length_2)+" element")
    print("The whole array is:")
    print(name)
    if(length!=1):
        print("The first row is:")
        print(name[0])
    print("----------------------------------------------------")

#
# @Function: Help you to match the result with 
#            the array index
#
def show_result(NN_list,show0save1=0):
    
    params        = eval(open("settings.txt").read())
    twitter_file  = params['twitter_path']
    list_length   = params['LIST_LENGTH']
    data_amount   = params['DATA_AMOUNT']
    save_path     = params['CLASSIFICATION']
    twitter_array = []

    with open(twitter_file,'r') as f:
        for line in f:
            twitter_array.append(line)
        f.close()
    
    NN_list_text = [[] for i in xrange(data_amount)]
    for data_index in xrange(data_amount):
        for list_index in xrange(list_length):
            index = NN_list[data_index][list_index]
            NN_list_text[data_index].append(twitter_array[index])

    if(show0save1==0):
        for data_index in xrange(data_amount):
            print('-----------------------------------')
            for list_index in xrange(list_length):
                print(NN_list_text[data_index][list_index])
    else:
        for data_index in xrange(data_amount):
            NN_list_text[data_index].append("----------------------------------")
        np.savetxt(save_path,NN_list_text,fmt='%s',delimiter='\n',newline ='\n')
        
            
