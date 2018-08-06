#
# @Params: vr_space      -> VR set must be prepared
#          feature_space -> Multiple input points
#
vr_space_dimension = 100
data_dimension = 784
def generate_vr_response(vr_space,feature_space){
    
    vr_space_length = len(vr_space)
    feature_space_length = len(feature_space)
    distance_space = np.zeros((vr_space_length,feature_space_length))
    response_space = np.zeros((vr_space_length,feature_space_length))

    for vr_index in xrange(vr_space_length):
        for graph_index in xrange(feature_space_length):
            distance_vr_graph = np.linalg.norm((vr_space[vr_index]-feature_space[feature_index]),ord=1)
            distanc_space[vr_index][graph_index] = distance_vr_graph
    
    
    average_distance_space = np.mean(distance_space,axis=1) 
    for vr_index in xrange(vr_space_length):
        average_distance = average_distance_space[vr_index]
        for graph_index in xrange(feature_space_length):
            response_vr_graph = np.exp(-((5 * distance_space[vr_index][graph_index])/average_distance)**0.7)
            response_space[vr_index][graph_index] = response_vr_graph

    return feature_space
}
