import theano

# theano.printing.debugprint(y)

def get_param(output_vars, named_only=False):
    '''
    Returns the shared parameters of a computation graph giev one or more
    output variables
    '''
    def get_shared(var):
        '''
        Recursion to get shared output_vars of the graph
        '''
        for node in var.get_parents():
            if isinstance(node, theano.tensor.sharedvar.TensorSharedVariable):

                if named_only:
                    if node.name != None:
                        params.append(node)
                else: 
                    params.append(node)
            else:
                get_shared(node)   
    params = []
    if not isinstance(output_vars, list):
        output_vars = [output_vars]
    for variable in output_vars:    
        get_shared(variable)    

    # Remove duplicates
    par_dict = {}
    unique_params = []
    for par in params[::-1]:
        if par not in par_dict:
            unique_params.append(par)
            par_dict[par] = True

    return unique_params 

# TODO: Function to search for parameter
