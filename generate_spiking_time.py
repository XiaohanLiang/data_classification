#                            
# @Function: This function was used to generate the spiking time of each neuron
#            which will be used as the input of PN layer in the next step.
#
# @Params: response_space -> response of vr to graph set must be given
#                            shape = (vr_count,graph_count)
# @Return: spiking_space  -> shape[0] = vr_count
#                            Each row represent the spiking time of each neuron
#                            
# @Config: Full rate, when vr_response==1
#          Set the spiking rate to be FULL_RATE Hz
params = eval(open("settings.txt").read())
FULL_RATE = params['FULL_SPIKING_RATE']

def generate_spiking_time(response_space,epoch=100):
    vr_count = len(response_space)
    graph_count = len(response_space[0])
    spiking_space = [[] for i in xrange(vr_count)]
    
    for graph_index in xrange(graph_count):
        base_time = graph_index * epoch
        end_time  = graph_index * epoch + epoch
        for vr_index in xrange(vr_count):
            
            rate = response_space[vr_index][graph_index]
            spiking_rate = FULL_RATE * rate
            spiking_interval = 1000 / spiking_rate
            count = 1
            
            while(base_time+count*spiking_interval<end_time):
                spiking_space[vr_index].append(base_time+count*spiking_interval)
                count = count+1
    return spiking_space

def generate_spiking_time_q(response_space,epoch=100):
    vr_count = len(response_space[0])
    graph_count = len(response_space)
    spiking_space = [[] for i in xrange(vr_count)]
    
    for graph_index in xrange(graph_count):
        print("Now dealing with index"+str(graph_index))
        base_time = graph_index * epoch
        end_time  = graph_index * epoch + epoch
        for vr_index in xrange(vr_count):
            print("Now dealing with vr_index "+str(vr_index) ) 
            print("Begin:"+str(base_time)+" End:"+str(end_time))
            rate = response_space[graph_index][vr_index]
            print("The rate is "+str(rate))
            if(rate==0):
                continue
            spiking_rate = FULL_RATE * rate
            spiking_interval = 1000 / spiking_rate
            count = 1
            
            while(base_time+count*spiking_interval<end_time):
                spiking_space[vr_index].append(base_time+count*spiking_interval)
                count = count+1
    return spiking_space
