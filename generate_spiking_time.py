#
# @Params: response_space -> response of vr to graph set must be given
#                            shape = (vr_count,graph_count)
#          epoch          -> Expose time used to train vr set must be given here
#

def generate_spiking_time(response_space,epoch){
    vr_count = len(response_space)
    graph_count = len(response_space[0])
    spiking_space = [[] for i in xrange(vr_count)]
    
    for graph_index in xrange(graph_count):
        base_time = graph_index * epoch
        end_time  = graph_index * epoch + epoch
        for vr_index in xrange(vr_count):
            
            rate = response_space[vr_index][graph_index]
            spiking_rate = 200*rate
            spiking_interval = 1000 / spiking_rate
            count = 1
            
            while(base_time+count*spiking_interval<end_time):
                spiking_space.append(base_time+count*spiking_interval)
                count = count+1
    
    return spiking_space
}
