import pandas
import numba
from numba import cuda
import numpy
import sys
import time
import csv
import math
import scipy.special
import scipy.stats
import scipy.interpolate




@cuda.jit(device=True)
def from_flat_index_binary(num_dims,flat_index,rslt):
    #Given the index into a flat array and the shape of the array, return the indices corresponding to
    #that that element in the flat array if it were shaped according to myshape
    dim_product=1
    for i in range(1,num_dims):
        dim_product=dim_product*2
    for i in range(num_dims-1):
        index=int(flat_index/dim_product)
        rslt[i]=index
        flat_index=flat_index-index*dim_product
        dim_product=dim_product/2
    rslt[num_dims-1]=flat_index
    return rslt


def from_flat_index_binary_cpu(num_dims,flat_index):
    #Given the index into a flat array and the shape of the array, return the indices corresponding to
    #that that element in the flat array if it were shaped according to myshape
    rslt=numpy.zeros(num_dims,dtype="uint8")
    dim_product=1
    for i in range(1,num_dims):
        dim_product=dim_product*2
    for i in range(num_dims-1):
        index=int(flat_index/dim_product)
        rslt[i]=index
        flat_index=flat_index-index*dim_product
        dim_product=dim_product/2
    rslt[num_dims-1]=flat_index
    return rslt


@cuda.jit("uint8[:](uint8[:],uint64,uint8[:])",device=True)
def from_flat_index(myshape,flat_index,result):
    #Given the index into a flat array and the shape of the array, return the indices corresponding to
    #that that element in the flat array if it were shaped according to myshape
    dim_product=1
    for i in range(1,len(myshape)):
        dim_product=dim_product*myshape[i]
    for i in range(len(myshape)-1):
        index=int(flat_index/dim_product)
        result[i]=index
        flat_index=flat_index-index*dim_product
        dim_product=dim_product/myshape[i+1]
    result[len(myshape)-1]=flat_index
    return result


@cuda.jit("uint8[:](uint64[:],uint64,uint8[:])",device=True)
def from_flat_index_quick(reverse_shape_prod,flat_index,result):
    #Given the index into a flat array and the reverse cumulative product of the shape of the array,
    #return the indices corresponding to
    # that element in the flat array if it were shaped according to the corresponding shape
    for i in range(len(reverse_shape_prod)):
        index=int(flat_index/reverse_shape_prod[i])
        result[i]=index
        flat_index=flat_index-index*reverse_shape_prod[i]
    result[len(reverse_shape_prod)]=flat_index
    return result


def reverse_prod(myshape):
    x=list(myshape[1:len(myshape)])+[1]
    x.reverse()
    y=numpy.array(list(numpy.cumprod(x)[len(x)-1:0:-1])).astype("uint64")
    return y



@cuda.jit(device=True)
def to_flat_index(myshape,indices):
    #Given the shape of an array and a tuple, find the index of that element if the array were flattened
    index=0
    multiplier=1
    for d in range(len(indices)-1,-1,-1):
        index=index+multiplier*indices[d]
        multiplier=multiplier*myshape[d]
    return index


@cuda.jit(device=True)
def to_flat_index_1(myshape,indices):
    #Given the shape of an array and a tuple, find the index of that element if the array were flattened
    index=int(0)
    multiplier=int(1)
    for d in range(len(myshape)-1,-1,-1):
        index=index+multiplier*indices[d]
        multiplier=multiplier*myshape[d]
    return index

@cuda.jit(device=True)
def to_flat_index_groups(group_shapes,group,indices):
    #Given the shape of an array and a tuple, find the index of that element if the array were flattened
    index=0
    multiplier=1
    for d in range(len(indices)-1,-1,-1):
        index=index+multiplier*indices[d]
        multiplier=multiplier*group_shapes[group,d]
    return index




@cuda.jit(device=True)
def myinsert(vector,length,value):
        #This is like bisect.bisect_left()
        #1D special case of insert_along_dimension
        n=length
        if value<=vector[0]:
            return 0
        else:
            if value>vector[n-1]:
                return n-1
            else:
                    done=False
                    left=0
                    right=n-1
                    test=int( (left+right)/2)
                    while right-left>1 and not done and vector[left]<vector[right]:
                        if value==vector[left]:
                            return left
                        if value==vector[right]:
                            return right
                        if value==vector[test]:
                            return test
                        if value<vector[test]:
                            right=test
                            test=int( (left+right)/2 )
                        if value>vector[test]:
                            left=test
                            test=int( (left+right)/2 )
                    rslt=left+(right-left)*(value-vector[left])/float(vector[right]-vector[left])
                    return rslt
    #                if right-left<=1:
     #                  return right
      #              if right-left>1:
       #                return int( (left+right)/2)

@cuda.jit
def create_grid(myshape,result,indices):
    i,j,k=cuda.grid(3)
    index= i<<16 | j<<6 | k

    if index<result.shape[0]:
        rslt=numba.cuda.local.array((100), numba.uint16)
        rslt=from_flat_index(myshape,index,rslt)
        for j in range(result.shape[1]):
            result[index,j]=rslt[indices[j]]

@cuda.jit(device=True)
#"uint16[:](float32[:,:],uint8[:],int16[:],uint32[:],int64[:],uint8[:,:],int64[:,:],float64[:],uint8,uint16[:])",device=True)
def thread_indices_to_coordinates(indices,flat_group_data,group_offsets,\
                                   group_lengths,group_shapes,structure,scale, \
                                    time_step,result):
        group_coordinates=numba.cuda.local.array((10,20), numba.uint8)
        for variable in range(structure.shape[0]):
            group=structure[variable,1]
            position_in_group=structure[variable,2]
            group_coordinates[group,position_in_group]=indices[variable]
            result[variable]=indices[variable]


        for variable in range(structure.shape[0]):
            space_code=structure[variable,7]
            variable_type=structure[variable,6]

            if variable_type==1:
                group=structure[variable,1]
                position_in_numeric_vars_list_for_group=structure[variable,3]
                group_coordinates[group,group_lengths[group]]=position_in_numeric_vars_list_for_group
                group_indices=group_coordinates[group,:(group_lengths[group]+1)]
                flat_index=group_offsets[group]+to_flat_index_groups(group_shapes,group,group_indices)
                #result[variable]=scale[variable]*float(var_value)
                result[variable]=flat_group_data[int(flat_index)]

        return result



@cuda.jit(device=True)
def exogenous_indices_to_coordinates(indices,flat_group_data,group_offsets,group_lengths,group_shapes,structure,scale,time_step,result):
        group_coordinates=numba.cuda.local.array((10,10), numba.uint8)
        for variable in range(structure.shape[0]):
            group=structure[variable,2]
            position_in_group=structure[variable,4]
            group_coordinates[group,position_in_group]=indices[variable]
            result[variable]=indices[variable]


        for variable in range(structure.shape[0]):
            space_code=structure[variable,-1]
            variable_type=structure[variable,-2]
            if variable_type==1 and space_code==3:
                group=structure[variable,2]
                position_in_numeric_vars_list_for_group=structure[variable,5]
                group_coordinates[group,group_lengths[group]]=position_in_numeric_vars_list_for_group
                group_indices=group_coordinates[group,:(group_lengths[group]+1)]
                flat_index=group_offsets[group]+to_flat_index_groups(group_shapes,group,group_indices)
                var_value=flat_group_data[int(flat_index)]
                result[variable]=var_value
        return result


@cuda.jit(device=True)
def project_coordinates(coordinates,flat_group_data,group_offsets,group_lengths,group_shapes,structure,scale,result):
        group_coordinates=numba.cuda.local.array((10,10), numba.uint8)
        state_space_length=0
        for variable in range(structure.shape[0]):
            group=structure[variable,2]
            position_in_group=structure[variable,4]
            type_code=structure[variable,6]
            space_code=structure[variable,7]
            if space_code==1:
                state_space_length=state_space_length+1
            if type_code==0:
                group_coordinates[group,position_in_group]=int(coordinates[variable])
                if space_code==1:
                   position_in_state_space=structure[variable,9]
                   result[position_in_state_space]=int(coordinates[variable])
            else:
                if space_code==1:
                   position_in_state_space=structure[variable,9]
                   result[position_in_state_space]=0
                group_coordinates[group,position_in_group]=0


        for variable in range(structure.shape[0]):
            space_code=structure[variable,-1]
            variable_type=structure[variable,-2]
            if variable_type==1 and space_code==1:
                value=coordinates[variable]/scale[variable]
                group=structure[variable,2]
                position_in_group=structure[variable,4]
                position_in_numeric_vars_list_for_group=structure[variable,5]
                dim_size=structure[variable,1]
                vector=numba.cuda.local.array((20), numba.uint16)
                for i in range(dim_size):
                    group_coordinates[group,position_in_group]=i
                    group_coordinates[group,group_lengths[group]]=position_in_numeric_vars_list_for_group
                    group_indices=group_coordinates[group,:(group_lengths[group]+1)]
                    flat_index=group_offsets[group]+to_flat_index_groups(group_shapes,group,group_indices)
                    var_value=flat_group_data[int(flat_index)]
                    vector[i]=var_value
                coord_index=myinsert(vector,dim_size,value)
                group_coordinates[group,position_in_group]=coord_index
                if space_code==1:
                   position_in_state_space=structure[variable,9]
                   result[position_in_state_space]=coord_index
        return result


@cuda.jit(device=True)
def get_probability(flat_prob_data,flat_prob_offsets,prob_lengths,prob_shapes,prob_structure,structure,indices):
        group_coordinates=numba.cuda.local.array((10,10), numba.uint8)
        for variable in range(structure.shape[0]):
            space_code=structure[variable,7]
            if space_code==3:
                prob_group=structure[variable,5]
                for i in range(prob_lengths[prob_group]):
                    group_coordinates[prob_group,i]=indices[prob_structure[prob_group,i]]
                group_coordinates[prob_group,prob_lengths[prob_group]]=indices[variable]
        prob_product=1.
        for variable in range(structure.shape[0]):
            space_code=structure[variable,7]
            if space_code==3:
                prob_group=structure[variable,5]
                prob_indices=group_coordinates[prob_group,:prob_lengths[prob_group]]
                prob_shape=prob_shapes[prob_group][:len(prob_indices)]
                flat_index=int(flat_prob_offsets[prob_group]+int(to_flat_index(prob_shape,prob_indices)))
                prob=flat_prob_data[flat_index]
                prob_product=prob_product*prob
        return prob_product



@cuda.jit(device=True)
def get_exogenous_probabilities(flat_cum_prob_data,flat_prob_offsets,prob_lengths,prob_shapes,prob_structure,structure,indices,result):
        group_coordinates=numba.cuda.local.array((10,10), numba.uint8)
        for variable in range(structure.shape[0]):
            space_code=structure[variable,7]
            if space_code==3:
                prob_group=structure[variable,5]
                for i in range(prob_lengths[prob_group]):
                    group_coordinates[prob_group,i]=indices[prob_structure[prob_group,i]]
                group_coordinates[prob_group,prob_lengths[prob_group]]=indices[variable]
        num_done=0
        for variable in range(structure.shape[0]):
            space_code=structure[variable,7]
            if space_code==3:
                prob_group=structure[variable,5]
                prob_indices=group_coordinates[prob_group,:prob_lengths[prob_group]]
                prob_shape=prob_shapes[prob_group][:len(prob_indices)]
                flat_index=int(flat_prob_offsets[prob_group]+int(to_flat_index(prob_shape,prob_indices)))
                prob=flat_cum_prob_data[flat_index]
                result[num_done]=prob
                num_done+=1





#@cuda.jit(device=True)
#def get_neighbors(state_space_indices,neighbor_variations,rslt):
#    for i in range(neighbor_variations.shape[0]):
##        for j in  range(len(state_space_indices)):
#                #rslt[i,j]=int(state_space_indices[j])+neighbor_variations[i,j]
#                intpart=int(state_space_indices[j])
#                frac_part=state_space_indices[j]-intpart
#                if frac_part>=.5:
#                    rslt[i,j]=intpart+1
#                else:
#                    rslt[i,j]=intpart
#    return neighbor_variations.shape[0]






@cuda.jit(device=True)
def get_neighbor_values(neighbors,flat_recursion_data,state_space_shape,result):
    for i in range(neighbors.shape[0]):
         index=0
         multiplier=1
         for j in range(neighbors.shape[1]-1,-1,-1):
            index=index+multiplier*neighbors[i,j]
            multiplier=multiplier*state_space_shape[j]
         result[i]=flat_recursion_data[index]
    return result


@cuda.jit(device=True)
def interpolate(neighbors,neighbor_values,point):
    estimate=0.0
    for i in range(neighbors.shape[0]):
        prob=1.0
        for j in range(neighbors.shape[1]):
            prob=prob*(1-abs(point[j]-neighbors[i,j]))
        estimate=estimate+neighbor_values[i]*prob
    return estimate



@cuda.jit(device=True)
def insert_along_dimension(flat_array,shape,indices,dimension,value):
    #This is like bisect.bisect_left()
    #Given an array specified by the data flat_array and the shape.
    #and given a starting point in the array indeces
    #and given what dimension in the array to search along
    #and given the value to insert at,
    #we return the index for that dimension where we would insert the value
    n=shape[dimension]
    myindices=indices
    vector=numba.cuda.local.array((100), numba.float32)
    for i in range(shape[dimension]):
        myindices[dimension]=i
        flat_index=to_flat_index(shape,myindices)
        vector[i]=flat_array[flat_index]
    if value<=vector[0]:
        return 0
    else:
        if value>vector[n-1]:
            return n
        else:
            left=0
            right=n-1
            test=int( (left+right)/2)
            while right-left>1 and vector[left]<vector[right]:
                if value==vector[left]:
                    return left
                if value==vector[right]:
                    return right
                if value==vector[test]:
                    return test
                if value<vector[test]:
                    right=test
                    test=int( (left+right)/2 )
                if value>vector[test]:
                    left=test
                    test=int( (left+right)/2 )
            if right-left<=1:
               return right
            if right-left>1:
               return int( (left+right)/2)






@cuda.jit(device=True)
def insert1(vector,length,value):
        #This is like bisect.bisect_left()
        #1D special case of insert_along_dimension
        n=length

        if value<=vector[0]:
            return 0
        else:
            if value>vector[n-1]:
                return n
            else:
                    done=False
                    left=0
                    right=n-1
                    test=int( (left+right)/2)
                    while right-left>1 and not done and vector[left]<vector[right]:
                        if value==vector[left]:
                            return left
                        if value==vector[right]:
                            return right
                        if value==vector[test]:
                            return test
                        if value<vector[test]:
                            right=test
                            test=int( (left+right)/2 )
                        if value>vector[test]:
                            left=test
                            test=int( (left+right)/2 )
                    if right-left<=1:
                       return right
                    if right-left>1:
                       return int( (left+right)/2)

#************************************************************
#Random number generator
MAX32 = numpy.uint32(0xffffffff)

@cuda.jit(device=True)
def cuda_rand_int(rand_result):
    x = rand_result[0]
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    rand_result[0] = x
    rand_int=x * 2685821657736338717
    rand_result[1]=rand_int
    return rand_result



#*******************************************************************************************

#simulate a Markov Chain
def mc_simulate(initial_dist,transition_matrix,num_sims, num_transitions):
    flat_matrix=transition_matrix.cumsum(1).flatten()
    initial_dist=initial_dist.cumsum()
    result=numpy.zeros((num_sims,num_transitions),dtype="uint8")
    num_states=len(initial_dist)
    seeds=numpy.random.randint(0,2**31,num_sims)
    mc_simulate_gpu[(128, 128, 8), (16, 8, 8)](initial_dist,flat_matrix,seeds,num_states,result)
    return result

#************************************************************************************


#GPU engine for Markov Chain simulation
@cuda.jit
def mc_simulate_gpu(initial_dist,flat_matrix,seeds,num_states,result):

    i,j,k=cuda.grid(3)
    index= i<<16 | j<<6 | k

    if index<result.shape[0]:
        #sample initial state


        #get random number seed and run random algorithm once to get random float
        seed=seeds[index]
        rand_int=numba.cuda.local.array((2), numba.uint32)
        rand_int[0]=seed
        rand_int=cuda_rand_int(rand_int)
        rand_float=float(float(MAX32 & rand_int[0]) / float(MAX32))


        #Find where random float would insert into cumulative prob initial_dist
        #That is initial state
        flat_array=initial_dist
        shape=numba.cuda.local.array((1), numba.uint32)
        shape[0]=num_states
        indices=numba.cuda.local.array((1), numba.uint32)
        dimension=0
        value=rand_float
        state=insert_along_dimension(flat_array,shape,indices,dimension,value)
        result[index,0]=state


        #setup for finding where random values insert into the transition prob matrix
        #TPM data is flat in flat_array and shape of it is in shape
        #dimension=1 becuase that is the dimension were searching/inserting along
        #state is the value of the first dimension becuase that is the row of the TPM we need to insert along
        shape=numba.cuda.local.array((2), numba.uint32)
        shape[0]=num_states
        shape[1]=num_states
        flat_array=flat_matrix
        indices=numba.cuda.local.array((2), numba.uint32)
        indices[1]=state
        dimension=1


        #simulate rest of markov chain. Each time we set the first index to the current state (row of TPM)
        #and search/insert along dimension 1 (columns of TPM)
        for i in range(1,result.shape[1]):
            x=int(453./10)
            rand_int=cuda_rand_int(rand_int)
            rand_float=float(float(MAX32 & rand_int[0]) / float(MAX32))
            value=rand_float
            state=insert_along_dimension(flat_array,shape,indices,dimension,value)
            result[index,i]=state
            indices[0]=state






@cuda.jit()
#"void(int8,float32[:],float32[:],uint32[:],int8[:],int8[:,:],int8[:,:],int64,int8[:],int8,uint16[:],uint32[:],uint8[:],int8[:,:],int8[:,:],float64[:],int8[:],uint8,uint8,float64[:,:])")
def update_state(batch,threads_per_batch,int_result,frac_result,prob_result,
               state_space_shape_reverse_prod,utility,
               constants,parameters,state_space_shape,flat_recursion_data,thread_shape_reverse_prod,neighbor_variations,
               exp_lookup_table,norm_lookup_table,offset,multiplier,
               flat_prob_data,flat_prob_offsets,prob_lengths,prob_shapes,
               prob_structure,num_threads,thread_shape,flat_group_data,
               group_offsets,group_lengths,group_shapes,structure,scale,
               exogenous_shape,state_space_length,num_neighbors,gpu_result):



    index=numba.uint32(cuda.grid(1))
    #i,j,k=cuda.grid(3)
    #index= i<<20 | j<<10 | k
    if index<threads_per_batch:
        grand_index=threads_per_batch*batch+index

        #get indices in (state,decision) space (lets also call it thread space) corresponding to index
        rslt=numba.cuda.local.array((20), numba.uint8)
        thread_indices=from_flat_index_quick(thread_shape_reverse_prod,grand_index,rslt)[:len(thread_shape)]



        #prepend time index to thread indices
        indices=numba.cuda.local.array((20), numba.int8)
        indices[0]=0
        for i in range(len(thread_indices)):
            indices[1+i]=thread_indices[i]
        indices=indices[:structure.shape[0]]


        #convert time+thread indices to time+thread_coordinates
        #i.e by converting numeric variables from their ordinal indexes to int16 scaled versions of their numeric values
        rslt1=numba.cuda.local.array((20), numba.uint16)
        int16_coordinates=thread_indices_to_coordinates(indices,flat_group_data,group_offsets,group_lengths,group_shapes,
                                                           structure,scale,0,rslt1)


        #apply offsets and multipliers to convert int16 versions of coordinates to actual float values
        indices=indices[:structure.shape[0]]
        coordinates=numba.cuda.local.array((20), numba.float64)
        for i in range(structure.shape[0]):
            type_code=structure[i,6]
            if type_code==1:
                coordinates[i]=offset[i]+multiplier[i]*float(int16_coordinates[i])
            else:
                coordinates[i]=float(int16_coordinates[i])
        coordinates=coordinates[:structure.shape[0]]


        #determine whether these coordinates are feasible according to the rules
        #feasible=is_feasible(coordinates,constants)

        if True:
        #if feasible>0:
            #Here's where we call the function to update the coordinates to what they should be in the next time step
            new_coordinates=numba.cuda.local.array((20), numba.float64)
            new_coordinates=update_state_space(coordinates,new_coordinates,parameters,constants)

            #temporary just for testing
            #new_coordinates=coordinates

            #project coordinates onto discrete state space and get indices
            newindex=numba.cuda.local.array((20), numba.float64)
            for i in range(state_space_length):
                type_code=structure[i,6]
                if type_code==1:
                    distribution_type=structure[i,8]
                    dim_size=structure[i,0]
                    #if distribution_type==0:
                    #    lookup=int(new_coordinates[i]/multiplier[i])
                    #else:
                    lookup=int( (new_coordinates[i]-offset[i])/multiplier[i] )
                    if lookup<0:
                        lookup=0
                    if lookup>65535:
                        lookup=65535
                    if distribution_type==0:
                        newindex[i]=exp_lookup_table[lookup]*dim_size-1
                    else:
                        newindex[i]=norm_lookup_table[lookup]*dim_size-1
                    if newindex[i]<0:
                        newindex[i]=0
                else:
                    newindex[i]=new_coordinates[i]
            state_space_indices=newindex[:state_space_length]


            #state_space_index=to_flat_index(state_space_shape,state_space_indices)
            #rslt2=numba.cuda.local.array((20), numba.uint8)
            #new_state_space_indices=from_flat_index_quick(state_space_shape_reverse_prod,state_space_index,rslt2)[:len(state_space_shape)]

            for i in range(1,7):
                    gpu_result[index,i]=state_space_indices[i]

            frac_index=0
            for i in range(1,state_space_length):
                type_code=structure[i,6]
                int_result[index,i-1]=int(state_space_indices[i])#int(new_state_space_indices[i-1])


                if type_code==1:
                    frac_result[index,frac_index]=int((state_space_indices[i]-int(state_space_indices[i]))*255)
                                             #int((new_state_space_indices[i]-int(new_state_space_indices[i]))*255)
                    frac_index=frac_index+1


            #Get probability for this exogenous space point
            prob=get_probability(flat_prob_data,flat_prob_offsets,prob_lengths,prob_shapes,prob_structure,structure,indices)
            prob_result[index]=int(65535*prob)





@cuda.jit()
def evaluate_decisions(time_step,batch,constants,
                       threads_per_batch,num_int_decisions,utility_shape,
                       state_int_batch,state_frac_batch,prob_batch,expectations_batch,values_by_numeric_decision_batch,
                       neighbor_variations,structure,state_space_shape,flat_recursion_data,exogenous_size,utility,
                       sum_expectation,gpu_result,mycount
                       ):

            index=cuda.grid(1)
            #Get state space indices
            state_space_indices=numba.cuda.local.array((20), numba.float32)
            state_space_length=state_int_batch.shape[1]+1
            state_space_indices[0]=time_step+1
            frac_index=0
            for i in range(1,state_space_length):
                state_space_indices[i]=state_int_batch[index,i-1]
                type_code=structure[i,6]
                if type_code==1:
                    state_space_indices[i]=state_space_indices[i]+float(state_frac_batch[index,frac_index])/255
                    frac_index=frac_index+1
            state_space_indices=state_space_indices[:state_space_length]
            prob=numba.float32(float(prob_batch[index])/65536)


            #Get neighbors in state space
            rslt5=numba.cuda.local.array((16,20), numba.uint8)
            for i in range(neighbor_variations.shape[0]):
                for j in  range(len(state_space_indices)):
                    rslt5[i,j]=int(state_space_indices[j])+neighbor_variations[i,j]

            neighbors=rslt5[:neighbor_variations.shape[0],:state_space_length]


            #get values of objective function (ie expected value at next time step) at neighbors
            rslt6=numba.cuda.local.array((32), numba.int32)#numba.float64)
            neighbor_values=get_neighbor_values(neighbors,flat_recursion_data,state_space_shape,rslt6)
            neighbor_values=neighbor_values[:neighbors.shape[0]]


            #Get interpolated value from neighboring points
            interp_val=interpolate(neighbors,neighbor_values,state_space_indices)

            #accumulate expectations
            state_decision_index=int(index/exogenous_size)
            numba.cuda.atomic.add(expectations_batch,state_decision_index,prob*interp_val)
            numba.cuda.syncthreads()


            #accumulate sums of exponentials for Emax calculation
            thread_in_block= index % exogenous_size
            if thread_in_block==0:
                state_numeric_decision_index=int(float(index)/(exogenous_size*num_int_decisions))
                val=expectations_batch[state_decision_index]
                beta=constants[4]
                util=utility[state_numeric_decision_index]
                expectations_batch[state_decision_index]=util+beta*val
                numba.cuda.atomic.add(sum_expectation,state_numeric_decision_index,val)
                numba.cuda.atomic.add(mycount,state_numeric_decision_index,1)
                numba.cuda.syncthreads()


                sumval=sum_expectation[state_numeric_decision_index]
                countval=mycount[state_numeric_decision_index]
                avgval=sumval/countval
                expval=numba.float32(math.exp(val-avgval))
                numba.cuda.atomic.add(values_by_numeric_decision_batch,\
                                       state_numeric_decision_index,\
                                          expval)
                numba.cuda.syncthreads()
                sumexpval=values_by_numeric_decision_batch[state_numeric_decision_index]

                #calculate utility(time_step)+beta*Value(time_step+1)
                if ( index % (exogenous_size*num_int_decisions) )==0:
                    logval=math.log(sumexpval)+avgval
                    value=beta*logval+util
                    values_by_numeric_decision_batch[state_numeric_decision_index]=value







@cuda.jit()
def forward_sim_get_max(max_exp,states,best_value,gpu_shape,state_decision_shape,structure,numeric_decision_reverse_prod,int_decision_reverse_prod,
                      rand_float,expectations,discounted_utility):

    i,j,k=cuda.grid(3)
    a=1
    if i<gpu_shape[0] and j<gpu_shape[1] and k<gpu_shape[2]:
        index=900*i+9*j+k
        instance=i
        numeric_decision_index=j
        int_decision_index=k
        num_int_decisions=len(int_decision_reverse_prod)+1

        #shared array for Gumbell Variates... should be the same for each thread
        gumbel_value=numba.cuda.shared.array(20,numba.float32)



        #determine values of decision variables this thread looks after
        numeric_decision_indices=numba.cuda.local.array(5,numba.uint8)[:len(numeric_decision_reverse_prod)+1]

        int_decision_indices=numba.cuda.local.array(5,numba.uint8)[:len(int_decision_reverse_prod)+1]
        numeric_decision_indices=from_flat_index_quick(numeric_decision_reverse_prod,numeric_decision_index,numeric_decision_indices)
        int_decision_indices=from_flat_index_quick(int_decision_reverse_prod,int_decision_index,int_decision_indices)
        numeric_indices_used=0
        int_indices_used=0

        for i in range(structure.shape[0]):
            if structure[i,7]==2 and structure[i,6]==1:
                states[instance,i]=numeric_decision_indices[numeric_indices_used]
                numeric_indices_used+=1
            if structure[i,7]==2 and structure[i,6]==0:
                states[instance,i]=int_decision_indices[int_indices_used]
                int_indices_used+=1


        if numeric_decision_index==0:
            #generate Gumbel(0,1) samples
            gumbel_value[int_decision_index]=-math.log(-math.log(rand_float[instance,int_decision_index]))
        numba.cuda.syncthreads()


        #every thread starts with same point in state space
        flat_index=to_flat_index_1(state_decision_shape[1:],states[instance,1:len(state_decision_shape)])
        value=expectations[flat_index]+gumbel_value[int_decision_index]
        discounted_utility[index]=numba.float32(value+10*max_exp)
        numba.cuda.atomic.max(best_value,instance,value+10*max_exp)



@cuda.jit()
def forward_sim_get_decision(time_step,states,best_value,gpu_shape,state_decision_shape,structure,
                            numeric_decision_reverse_prod,int_decision_reverse_prod,
                            result,argmax):
     i,j,k=cuda.grid(3)
     instance=i
     numeric_decision_index=j
     int_decision_index=k
     if i<gpu_shape[0] and j<gpu_shape[1] and k<gpu_shape[2]:
         big_index=gpu_shape[1]*gpu_shape[2]*i+gpu_shape[2]*j+k
         little_index=gpu_shape[2]*j+k
         instance=i
         if result[big_index]==best_value[instance]:
             numba.cuda.atomic.max(argmax,instance,little_index)
         numba.cuda.syncthreads()
         if little_index==argmax[instance]: #ie this thread has the best decision

             state=states[instance,:]



             #determine values of decision variables this thread looks after
             numeric_decision_indices=numba.cuda.local.array(5,numba.uint8)[:len(numeric_decision_reverse_prod)+1]
             int_decision_indices=numba.cuda.local.array(5,numba.uint8)[:len(int_decision_reverse_prod)+1]
             indices=numba.cuda.local.array(20,numba.uint8)[:(len(state)+len(numeric_decision_indices)+len(int_decision_indices))]
             indices=indices[:structure.shape[0]]
             numeric_decision_indices=from_flat_index_quick(numeric_decision_reverse_prod,numeric_decision_index,numeric_decision_indices)
             int_decision_indices=from_flat_index_quick(int_decision_reverse_prod,int_decision_index,int_decision_indices)
             numeric_indices_used=0
             int_indices_used=0

             indices[0]=time_step
             for i in range(1,len(state)):
                 indices[i]=state[i]

             for i in range(structure.shape[0]):
                 if structure[i,7]==2 and structure[i,6]==1:
                     states[instance,i]=numeric_decision_indices[numeric_indices_used]
                     numeric_indices_used+=1
                 if structure[i,7]==2 and structure[i,6]==0:
                     states[instance,i]=int_decision_indices[int_indices_used]
                     int_indices_used+=1

             #at this point indeices has all the state except for the exogenous variables

@cuda.jit()
def forward_sim_get_exogenous(states,exogenous_shape_reverse_prod,space_code,rand_floats,
                                flat_cum_prob_data,flat_prob_offsets,prob_lengths,prob_shapes,prob_structure,
                                structure,exogenous):

    i,j=cuda.grid(2)
    instance=i
    exogenous_index=j

    num_exogenous_vars=len(exogenous_shape_reverse_prod)+1


    #fill in exogenous vars for this thread
    exogenous_indices=numba.cuda.local.array(10,numba.uint8)
    from_flat_index_quick(exogenous_shape_reverse_prod,exogenous_index,exogenous_indices)
    indices=numba.cuda.local.array(20,numba.uint8)
    exogenous_done=0
    for i in range(len(space_code)):
        if space_code[i]<3:
            indices[i]=states[instance,i]
        else:
            indices[i]=exogenous_indices[exogenous_done]
            exogenous_done+=1

    #replace the exogenous variables with the corresponding cumulative probability for this thread
    cumprobs=numba.cuda.local.array(10,numba.float64)
    get_exogenous_probabilities(flat_cum_prob_data,flat_prob_offsets,prob_lengths,prob_shapes,prob_structure,structure,indices,cumprobs)

    #add one to exogenous state variable where probabilities are less than the random float
    #In the end, that state variable will be equal to the number of threads with cumprob < the random number
    #i.e it is the sample for that variable according to the cumulative probabilities
    for i in range(num_exogenous_vars):#range(num_exogenous_vars):
        if cumprobs[i]<rand_floats[instance,i]:
            numba.cuda.atomic.max(exogenous,(instance,i),exogenous_indices[i]+1)
    numba.cuda.syncthreads()

    if exogenous_index==0:
        exogenous_done=0
        for i in range(structure.shape[0]):
            if space_code[i]==3:
                states[instance,i]=exogenous[instance,exogenous_done]
                exogenous_done+=1


@cuda.jit()
def forward_sim_update_state(states,flat_group_data,group_offsets,group_lengths,group_shapes,
                                                           structure,scale,time_step,offset,multiplier,constants,parameters,
                                                            state_space_length,exp_lookup_table,norm_lookup_table,result):

        instance=cuda.grid(1)
        indices=states[instance,:]
        time_step=indices[0]
        rslt1=numba.cuda.local.array((20), numba.uint16)
        int16_coordinates=thread_indices_to_coordinates(indices,flat_group_data,group_offsets,group_lengths,group_shapes,
                                                           structure,scale,time_step,rslt1)


        #apply offsets and multipliers to convert int16 versions of coordinates to actual float values
        indices=indices[:structure.shape[0]]
        coordinates=numba.cuda.local.array((20), numba.float64)
        for i in range(structure.shape[0]):
            type_code=structure[i,6]
            if type_code==1:
                coordinates[i]=offset[i]+multiplier[i]*float(int16_coordinates[i])
            else:
                coordinates[i]=float(int16_coordinates[i])
        coordinates=coordinates[:structure.shape[0]]

        for i in range(structure.shape[0]):
            result[instance,i]=coordinates[i]


        #Here's where we call the function to update the coordinates to what they should be in the next time step
        new_coordinates=numba.cuda.local.array((20), numba.float64)
        new_coordinates=update_state_space(coordinates,new_coordinates,parameters,constants)


        #project coordinates onto discrete state space and get indices
        newindex=numba.cuda.local.array((20), numba.float64)
        for i in range(state_space_length):
            type_code=structure[i,6]
            if type_code==1:
                distribution_type=structure[i,8]
                dim_size=structure[i,0]
                lookup=int( (new_coordinates[i]-offset[i])/multiplier[i] )
                if lookup<0:
                    lookup=0
                if lookup>65535:
                    lookup=65535
                if distribution_type==0:
                    newindex[i]=exp_lookup_table[lookup]*dim_size-1
                else:
                    newindex[i]=norm_lookup_table[lookup]*dim_size-1
                if newindex[i]<0:
                    newindex[i]=0
            else:
                newindex[i]=new_coordinates[i]
        state_space_indices=newindex[:state_space_length]
        for i in range(state_space_length):
            states[instance,i]=state_space_indices[i]



@cuda.jit()
def mytest(result):
    i,j,k=cuda.grid(3)
    x=numba.cuda.shared.array(9,numba.int32)
    if j==0 and k==0:
        for p in range(9):
            x[p]=p
    #numba.cuda.atomic.max(result,0,i)
    #numba.cuda.atomic.max(result,1,j)
    #numba.cuda.atomic.max(result,2,k)
    numba.cuda.syncthreads()
    result[i,j,k]=x[k]

result=numpy.zeros((1000,100,9))
mytest[(1000,1,1),(1,100,9)](result)









############################################################################################






def get_utility(constants,parameters,structure,offset,multiplier,flat_grid_data):
     consumption_utility_multiplier=parameters[18]
     labor_disutility=parameters[0]

     labor_elasticity=parameters[1]
     intertemp_elasticity=constants[1]
     max_hours_per_year=constants[0]
     group_vars=numpy.where(structure[:,1]==0)[0]
     group=structure[:,1]
     type_code=structure[:,6]
     group_shape=tuple(structure[group_vars,0])+(sum(numpy.logical_and(group==0,type_code==1)),)
     data=flat_grid_data[:numpy.prod(group_shape)].reshape(group_shape)
     wage=offset[5]+multiplier[5]*data[0,0,0,0,:,:,:,:,1].astype("float64")
     frac_savings=offset[7]+multiplier[7]*data[0,0,0,0,:,:,:,:,2].astype("float64")

     #just temporarily
     #frac_savings[:]=0
     #wage[:]=20
     #print consumption_utility_multiplier
     #print intertemp_elasticity
     #print labor_disutility
     #print labor_elasticity
     #print max_hours_per_year
     #hours=max_hours_per_year*0.2857037
     #consumption=20*hours

#     utility=consumption_utility_multiplier*consumption**(1-intertemp_elasticity)/(1-intertemp_elasticity) \
#                   -labor_disutility*(hours/max_hours_per_year)**(1+labor_elasticity)/(1+labor_elasticity)

     hours=offset[8]+multiplier[8]*data[0,0,0,0,:,:,:,:,3].astype("float64")


     consumption=wage*hours*(1-frac_savings)
     utility=consumption_utility_multiplier*consumption**(1-intertemp_elasticity)/(1-intertemp_elasticity) \
                   -labor_disutility*(hours/max_hours_per_year)**(1+labor_elasticity)/(1+labor_elasticity)
     utility=numpy.tile(utility.flatten(),structure[6,0])
     new_shape=(structure[6,0],)+group_shape[4:8]
     utility=utility.reshape(new_shape).transpose((1,2,0,3,4)).flatten()
     utility_shape=numpy.ascontiguousarray(structure[4:9,0])
     return utility,utility_shape#,consumption,hours/max_hours_per_year,wage,frac_savings


def get_recursion_data(constants,parameters,structure,offset,multiplier,flat_grid_data,retirement_utility_lookup):
     consumption_utility_multiplier=parameters[18]
     labor_disutility=parameters[0]
     labor_elasticity=parameters[1]
     intertemp_elasticity=constants[1]
     max_hours_per_year=constants[0]
     group_vars=numpy.where(structure[:,1]==0)[0]
     group=structure[:,1]
     type_code=structure[:,6]
     group_shape=tuple(structure[group_vars,0])+(sum(numpy.logical_and(group==0,type_code==1)),)
     data=flat_grid_data[:numpy.prod(group_shape)].reshape(group_shape)
     assets=offset[4]+multiplier[4]*data[0,:,:,:,:,:,0,0,0]
     assets=assets.flatten().astype("float64")
     assets=assets.reshape((len(assets),1))*numpy.ones((1,structure[6,0]))
     assets=assets.flatten()
    # consumption=assets
     #hours=0
     print(retirement_utility_lookup.shape,max(assets))
     #boundary_utility=consumption_utility_multiplier*consumption**(1-intertemp_elasticity)/(1-intertemp_elasticity) #\
     boundary_utility=consumption_utility_multiplier*retirement_utility_lookup[numpy.floor(assets).astype('uint32')]
     initial_recursion_data=numpy.zeros( (structure[0,0],len(boundary_utility)) ,dtype='float32')
     initial_recursion_data[-1,:]=boundary_utility
     initial_recursion_data=initial_recursion_data.flatten()
     initial_recursion_shape=structure[range(7),0]
     return initial_recursion_data,initial_recursion_shape


def get_probabilities(structure,constants,parameters,flat_grid_data):
    skill_persistance=parameters[7]
    price_shock_sd=math.sqrt(parameters[6])
    num_z_quantiles=structure[-1,0]
    z_quantiles=(offset[-1]+multiplier[-1]*flat_grid_data[-num_z_quantiles:].astype("float64"))\
                                       *price_shock_sd/math.sqrt(1-skill_persistance)
    zprobs=1-.5*scipy.special.erfc(z_quantiles/math.sqrt(2))
    zfinal_probs=numpy.array([.001]+list(zprobs[:-1]+.5*numpy.diff(zprobs))+[.999])
    zfinal=scipy.stats.norm.isf(1-zfinal_probs).reshape( (1,len(zfinal_probs)) )
    zinitial=z_quantiles.reshape( (len(z_quantiles),1) )
    comp_probs= .5*scipy.special.erfc( (zfinal-zinitial)/price_shock_sd/math.sqrt(2)  )
    Probs=-numpy.diff(comp_probs)
    cumProbs=Probs.cumsum(1)
    return Probs,cumProbs

def get_flat_probabilities(structure,constants,parameters,flat_grid_data,flat_prob_new_job,flat_pp_prob,flat_cum_prob_new_job,flat_pp_cum_prob):
    ProbMatrix,cumProbMatrix=get_probabilities(structure,constants,parameters,flat_grid_data)
    rslt=numpy.hstack((flat_prob_new_job,flat_pp_prob,ProbMatrix.flatten()))
    cumrslt=numpy.hstack((flat_cum_prob_new_job,flat_pp_cum_prob,cumProbMatrix.flatten()))
    return rslt,cumrslt


@cuda.jit(device=True)
def is_feasible(coordinates,constants):
    assets=coordinates[4]
    wage=coordinates[5]
    frac_savings=coordinates[7]
    hours=coordinates[8]
    interest_rate=constants[2]
    tax_rate=constants[6]
    min_assets=constants[5]
    new_assets=assets*(1+interest_rate)+wage*hours*(1-tax_rate)*frac_savings
    if new_assets>=min_assets:
        return 1
    else:
        return 0

#@cuda.jit(device=True)
def pack_bits(indices,bit_widths,num_bytes,length,rslt):
    x=0
    for i in range(length-1):
        x=x+indices[i]
        x=x<<bit_widths[i+1]
    x=x+indices[length-1]
    print("x",x)
    for i in range(num_bytes-1):
        rslt[num_bytes-i-1]=x & 255
        x=x>>8
    rslt[0]=x & 255

    print ("rslt",rslt)
    return rslt

#@cuda.jit(device=True)
def unpack_bits(y,bit_widths,bit_width_powers,num_bytes,length,rslt):
    x=0
    print ("again result",y)
    for i in range(num_bytes-1):
        x=x+y[i]
        x=x<<8
    x=x+y[num_bytes-1]
    print ("x again",x)
    for i in range(length-1,-1,-1):
        rslt[i]=x & (bit_width_powers[i]-1)
        x=x>> bit_widths[i]
    return rslt

#@cuda.jit(device=True)
def pack_state_space(indices,bit_widths,bit_width_powers,num_bytes,length,num_float,rslt):
#    new_indices=numba.cuda.local.array((20), numba.uint8)
    new_indices=numpy.zeros((20),dtype="uint8")
    for i in range(len(indices)):
        new_indices[i]=int(indices[i])
    for i in range(num_float):
        new_indices[len(indices)+i]=int( (indices[length-num_float+i]-int(indices[length-num_float+i]))*bit_width_powers[length+i] )
    return pack_bits(new_indices,bit_widths,num_bytes,length+num_float,rslt)


#@cuda.jit(device=True)
def unpack_state_space(x,bit_widths,bit_width_powers,num_bytes,length,num_float,rslt2):
#    rslt1=numba.cuda.local.array((20), numba.uint8)
    rslt1=numpy.zeros(20,dtype="uint8")
    rslt1=unpack_bits(x,bit_widths,bit_width_powers,num_bytes,length+num_float,rslt1)
    print("rslt1",rslt1)
    for i in range(length):
        rslt2[i]=rslt1[i]
    for i in range(num_float):
        print("float",length-num_float+i,length+i,float(rslt1[length+i])/bit_width_powers[length+i])
        rslt2[length-num_float+i]=rslt1[length-num_float+i]+float(rslt1[length+i])/bit_width_powers[length+i]
    return rslt2




@cuda.jit(device=True)
def update_state_space(coordinates,new_coordinates,parameters,constants):

    #Get parameters (varying these for fit)
    skill_price_constant=parameters[8]
    skill_price_by_job=numba.cuda.local.array((9), numba.float64)
    skill_price_by_job[0]=0
    for i in range(1,9):
        skill_price_by_job[i]=parameters[i+9]
    pp_premium=parameters[9]
    specific_skill_depreciation=parameters[5]
    general_skill_depreciation=parameters[4]
    learning_complementarity=parameters[3]

    #Get constants
    human_capital_accumulation_rate=constants[3]
    hours_per_year=constants[0]
    interest_rate=constants[2]
    tax_rate=constants[6]
    min_assets=constants[5]

    #Get current coordinates
    age=coordinates[0]
    job=int(coordinates[1])
    pp=coordinates[2]
    starting_new_job=coordinates[3]
    assets=coordinates[4]
    wage=coordinates[5]
    z=coordinates[6]
    frac_savings=coordinates[7]
    hours=coordinates[8]
    job_applied_to=int(coordinates[9])
    got_new_job=coordinates[10]
    new_pp=coordinates[11]
    new_z=coordinates[12]


    #update coordinates to next year
    skill_price=skill_price_constant+pp_premium*pp+skill_price_by_job[job]+specific_skill_depreciation*starting_new_job+z
    human_capital=wage/skill_price
    new_human_capital=human_capital_accumulation_rate*hours/hours_per_year*human_capital**learning_complementarity \
                 +(1-general_skill_depreciation)*human_capital
    new_assets=assets*(1+interest_rate)+wage*hours*(1-tax_rate)*frac_savings
    if got_new_job>0:
        new_job=job_applied_to
        new_pp=new_pp
        new_starting_new_job=1
    else:
        new_job=job
        new_pp=pp
        new_starting_new_job=0
    new_skill_price=skill_price_constant\
                     +pp_premium*new_pp\
                     +skill_price_by_job[new_job]\
                     +specific_skill_depreciation*new_starting_new_job\
                     +new_z
    new_wage=new_skill_price*new_human_capital


    #plug new coordinates into output
    new_coordinates[0]=coordinates[0]+1
    new_coordinates[1]=new_job
    new_coordinates[2]=new_pp
    new_coordinates[3]=new_starting_new_job
    new_coordinates[4]=new_assets
    new_coordinates[5]=new_wage
    new_coordinates[6]=new_z
    return new_coordinates





#**********************************************************************************************************************

#Read in structure of variables and spaces etc.
lines=list(csv.reader(open('data/structure.csv')))
headers=lines[0]
lines=lines[1:]
header2col=dict(zip(headers,range(len(headers))))
intcols=["dimension_size",
         "group",
         "position_in_group",
         "position_in_sequence_of_numeric_variables_in_group",\
          "position_in_state_space",
         "probability_group",
          "type_code","space_code","distribution_type"]
structure=numpy.ascontiguousarray(numpy.array([ [int(line[header2col[col]]) for col in intcols] for line in lines])).astype("int8")
varnames=[x[0].rstrip() for x in lines]
dim_size=structure[:,0]
group_code=structure[:,1]
group_position=structure[:,2]
numeric_position=structure[:,3]
state_space_position=structure[:,4]
probability_group=structure[:,5]
type_code=structure[:,6]
space_code=structure[:,7]
offset=numpy.ascontiguousarray(numpy.array([float(line[header2col["offset"]]) for line in lines])).astype("float64")
multiplier=numpy.ascontiguousarray(numpy.array([float(line[header2col["multiplier"]]) for line in lines])).astype("float64")
scale=numpy.ascontiguousarray(numpy.array([float(line[header2col["scale"]]) for line in lines])).astype("float64")
state_space_length=numpy.sum(space_code==1)
state_space_shape=dim_size[space_code==1]
state_space_vars=numpy.ascontiguousarray(numpy.array([v for v in range(structure.shape[0]) if space_code[v]==1]))
state_space_is_numeric=numpy.ascontiguousarray(numpy.array(type_code[state_space_vars]==1,dtype="bool_"))
state_space_shape_reverse_prod=reverse_prod(state_space_shape)

#get variations which will generate neighbors in state space
num_state_space_numeric_vars=sum(state_space_is_numeric)
num_neighbors=numpy.uint8(2**num_state_space_numeric_vars)
neighbor_numeric_variations=numpy.zeros( (num_neighbors,num_state_space_numeric_vars),dtype="uint8")
for i in range(num_neighbors):
    neighbor_numeric_variations[i,:]=from_flat_index_binary_cpu(num_state_space_numeric_vars,i)
neighbor_variations=numpy.zeros((num_neighbors,len(state_space_shape)),dtype="uint8")
neighbor_variations[:,state_space_is_numeric]=neighbor_numeric_variations

state_space_competition=numpy.ascontiguousarray(numpy.array(state_space_shape,dtype="float32"))
state_space_winner=numpy.ascontiguousarray(numpy.array(state_space_shape,dtype="uint32"))


#get size and shape of exogenous variables
exogenous_shape=numpy.ascontiguousarray(numpy.array([dim_size[i] for i in range(structure.shape[0])
                                                                                if space_code[i]==3])).astype("int8")
exogenous_size=numpy.product(exogenous_shape)

#Get data group lengths
num_groups=len(set(group_code))
group_length=numpy.ascontiguousarray(numpy.bincount(group_code[group_code>=0]))


#get structure of thread space
thread_vars=range(1,structure.shape[0])
thread_shape=numpy.ascontiguousarray(dim_size[thread_vars]).astype("uint8")
num_threads=numpy.product(thread_shape) #product of state and decision space sizes
thread_shape_reverse_prod=reverse_prod(thread_shape)


#Read in metadata on input data
lines=list(csv.reader(open("data_groups.csv")))
group_metadata=numpy.ascontiguousarray(numpy.array([[int(x) for x in line] for line in lines[1:]]))
group_lengths=numpy.ascontiguousarray(group_metadata[:,1]).astype("uint8")
group_metadata=group_metadata[:,2:]

#Count the number of numeric variables in each group
num_numeric_by_group=numpy.bincount(group_code[type_code==1])


#Get shape  (i.e. where data starts) for each group of data
#We will store all of this data in a flat array
group_shapes=numpy.ascontiguousarray(numpy.ones(group_metadata.shape)).astype("int8")
group_shapes[numpy.where(group_metadata>=0)]=dim_size[group_metadata[numpy.where(group_metadata>=0)]]
group_shapes=numpy.hstack( (group_shapes,numpy.ones((group_shapes.shape[0],1))) )
for group in range(group_shapes.shape[0]):
    group_shapes[group,group_length[group]]=num_numeric_by_group[group]

#Get structure of probability data
lines=list(csv.reader(open("prob_structure.csv")))
prob_structure=numpy.ascontiguousarray(numpy.array([[int(x) for x in line] for line in lines[1:]]))
prob_lengths=numpy.ascontiguousarray(prob_structure[:,1]).astype("int8")
prob_structure=numpy.ascontiguousarray(prob_structure[:,2:]).astype("int8")


#Get shape  (i.e. where data starts) for each group of group of probability data
#We will store all of this data in a flat array
prob_shapes=numpy.ascontiguousarray(numpy.ones(prob_structure.shape)).astype("int8")
prob_shapes[numpy.where(prob_structure>=0)]=dim_size[prob_structure[numpy.where(prob_structure>=0)]]
flat_prob_offsets=numpy.ascontiguousarray(numpy.array([0]+list(numpy.product(prob_shapes,1).cumsum()[:-1]))).astype("uint32")

#Get the number of elements in the flat version of each group
#Note the size is the product of the dimensions multiplied by the number of numeric variables
#for example if we have 4 variable (a,b,c,d) with dimension sizes (3,5,2,6) and
# b and c are numeric, then we need an array of shape (3,5,2,6,2) to store it, where the last dimension stores
#the value of the numeric variables
group_data_sizes=numpy.product(group_shapes,1)*(num_numeric_by_group>0)

#get the offset... ie where the data starts in the flat version, for each group
group_offsets=numpy.ascontiguousarray(numpy.array([0]+list(group_data_sizes.cumsum()[:len(group_data_sizes)-1]))).astype("uint32")


#Read in data which defines grid
flat_grid_data=numpy.load("data/flat_grid_data.npy").astype("uint16")
grid=flat_grid_data[:-20].reshape((31,9,2,2,10,10,10,10,4))
asset_values=list(set(grid[:,:,:,:,:,:,:,:,0].flatten()))

#read in probability data for getting a new job
lines=list(csv.reader(open("transition_probs.csv")))
headers=lines[0]
header2col=dict(zip(headers,range(len(headers))))
lines=lines[1:]
probs=numpy.array([float(line[header2col["lambda_e"]]) for line in lines]).reshape((len(lines),1))
probs=probs*numpy.ones((1,len(lines)))
Probs=numpy.zeros((len(lines),len(lines),2))
Probs[:,:,0]=1-probs
Probs[:,:,1]=probs
flat_prob_new_job=Probs.flatten()
flat_cum_prob_new_job=Probs.cumsum(2).flatten()

#read in probability data for a new job being performance pay
lines=list(csv.reader(open("share_pp_by_age.csv")))
headers=lines[0]
header2col=dict(zip(headers,range(len(headers))))
lines=lines[1:]
age=numpy.array([int(line[header2col["age"]]) for line in lines])
job=numpy.array([int(line[header2col["jobgrp"]]) for line in lines])
pp_prob=numpy.array([float(line[header2col["pp_age_jobgrp"]]) for line in lines])
triples=zip(age,job,pp_prob)
triples.sort()
pp_prob=numpy.zeros((len(triples),2))
pp_prob[:,1]=numpy.array([triple[2] for triple in triples])
pp_prob[:,0]=1-pp_prob[:,1]
flat_pp_prob=pp_prob.flatten()
flat_pp_cum_prob=pp_prob.cumsum(1).flatten()

#Initialize the data (flat_recursion_data) that we're going to iterate on
state_space_size=numpy.product(state_space_shape)

#******************************************************************************


#Initialize array to store expected value of each decision
num_time_steps=structure[0,0]
num_state_decision=int(num_threads/exogenous_size)
expectations=numpy.ascontiguousarray(numpy.zeros( (num_time_steps,num_state_decision/num_time_steps),dtype="float32"))
num_int_decisions=numpy.product([dim_size[i] for i in range(structure.shape[0]) if type_code[i]==0 and space_code[i]==2])
numeric_space_size=numpy.product([dim_size[i] for i in range(structure.shape[0]) if type_code[i]==1 and space_code[i]<=2])




flat_group_data=flat_grid_data

#Load lookup tables for exponential and normal distributions
exp_lookup_table=numpy.load("data/exp_lookup_table.npy").astype("float64")
norm_lookup_table=numpy.load("data/norm_lookup_table.npy").astype("float64")

num_batches=int(numpy.product([dim_size[i] for i in range(structure.shape[0]) if type_code[i]==0 and space_code[i]==1 and i>0]) )

threads_per_batch=int(num_threads/num_batches)
num_numeric_state=numpy.product([dim_size[i] for i in range(structure.shape[0]) if type_code[i]==1 and space_code[i]==1])
num_numeric_decision=numpy.product([dim_size[i] for i in range(structure.shape[0]) if type_code[i]==1 and space_code[i]==2])



state_space_ints=numpy.array([i for i in range(structure.shape[0]) if type_code[i]==0 and space_code[i]==1 and i>0])
state_space_floats=numpy.array([i for i in range(structure.shape[0]) if type_code[i]==1 and space_code[i]==1 and i>0])
batch_result_int=numba.cuda.device_array((threads_per_batch,state_space_length-1),dtype="uint8")
batch_result_frac=numba.cuda.device_array((threads_per_batch,len(state_space_floats)),dtype="uint8")
batch_result_prob=numba.cuda.device_array((threads_per_batch),dtype="uint16")
update_state_result_int=numpy.zeros((num_batches,threads_per_batch,state_space_length-1),dtype="uint8")
update_state_result_frac=numpy.zeros((num_batches,threads_per_batch,len(state_space_floats)),dtype="uint8")
update_state_result_prob=numpy.zeros((num_batches,threads_per_batch),dtype="uint16")
value=numpy.zeros((num_batches,num_numeric_state),dtype="float32")



threads_per_block=exogenous_size*num_int_decisions
blocks_per_batch=threads_per_batch/threads_per_block

state_numeric_decision_shape_batch=numpy.array([structure[i,0] for i in range(structure.shape[0]) if i>0 and space_code[i]<=2 and type_code[i]==1])
reverse_prod_state_numeric_decision_shape_batch=reverse_prod(state_numeric_decision_shape_batch)



#******************************************************************************

#Get Initial States


#get initial (job,pp,wage)
lines=list(csv.reader(open('data/initial_states.csv')))
headers=lines[0]
header2col=dict(zip(headers,range(len(headers))))
lines=lines[1:]
job=numpy.array([int(line[header2col['job']]) for line in lines])
pp=numpy.array([int(line[header2col['pp']]) for line in lines])
wage=numpy.array([float(line[header2col['wage']]) for line in lines])

#add the rest of the state variables
assets=numpy.array([22000 for i in range(len(wage))])
new_job=numpy.zeros(wage.shape)
Z=numpy.zeros(wage.shape)

#get grid values for assets and wages
num_years=structure[0,0]
group_0_size=numpy.product([structure[i,0] for i in range(structure.shape[0]) if group_code[i]==0])
asset_values=list(set(flat_grid_data[:4*group_0_size].reshape((num_years,group_0_size/num_years,4))[0,:,0]))
wage_values=list(set(flat_grid_data[:4*group_0_size].reshape((num_years,group_0_size/num_years,4))[0,:,1]))
asset_values.sort()
wage_values.sort()
asset_values=offset[4]+multiplier[4]*numpy.array(asset_values)
wage_values=offset[5]+multiplier[5]*numpy.array(wage_values)

#interpolate to get indices for initial assets and wages
asset_indices=numpy.round(numpy.interp(assets,asset_values,range(len(asset_values))))
wage_indices=numpy.round(numpy.interp(wage,wage_values,range(len(wage_values))))


#put together intial indices
initial_states=numpy.ascontiguousarray(numpy.zeros((len(wage_indices),structure.shape[0]),dtype='uint8'))
initial_states[:,1]=job-1
initial_states[:,2]=pp
initial_states[:,4]=asset_indices
initial_states[:,5]=wage_indices


numeric_decision_shape=numpy.array([structure[i,0] for i in range(structure.shape[0]) if space_code[i]==2 and type_code[i]==1])
numeric_decision_reverse_prod=reverse_prod(numeric_decision_shape)
int_decision_shape=numpy.array([structure[i,0] for i in range(structure.shape[0]) if space_code[i]==2 and type_code[i]==0])
int_decision_reverse_prod=reverse_prod(int_decision_shape)
print(numeric_decision_shape)
print(int_decision_shape)

state_decision_shape=numpy.array([structure[i,0] for i in range(structure.shape[0]) if space_code[i]<3])

states=initial_states
for i in range(100):
    states=numpy.vstack( (states,initial_states))
initial_states=numpy.copy(states)
gpu_shape=numpy.array([states.shape[0],numpy.product(numeric_decision_shape),numpy.product(int_decision_shape)])

#****************************************************************************************************
#Retirement model

lines=list(csv.reader(open('data/retirement_utility_lookup.csv')))[1:]
assets=numpy.array([int(line[0]) for line in lines])
utility=numpy.array([float(line[1]) for line in lines])
spline=scipy.interpolate.splrep(assets, utility, s=0)
xnew=numpy.arange(max(assets)+1)
retirement_utility_lookup=scipy.interpolate.splev(xnew,spline,der=0)

#************************************************************************************************
#************************************************************************************************************
#************************************************************************************************************
initial_parameters=numpy.array([34,.5,.7,0,-.05,-.08,.03,.97,2.72,.28,.21,-.011,-.17,.13,-.01,-.51,-.16,-.22,2000])
constants=numpy.array([5110,1.5,.04,1,.98,22000,.28])


#*************************************************************************************************************



def evaluate_parameter_set(parameters):
    global space_code

    flat_prob_data,flat_cum_prob_data=get_flat_probabilities(structure,constants,parameters,flat_grid_data,flat_prob_new_job,flat_pp_prob,flat_cum_prob_new_job,flat_pp_cum_prob)

    #Initialize the data (flat_recursion_data) that we're going to iterate on
    state_space_size=numpy.product(state_space_shape)
    flat_recursion_data,flat_recursion_data_shape=get_recursion_data(constants,parameters,structure,offset,multiplier,
                                                                                flat_grid_data,retirement_utility_lookup)

    #Getting data which is a function of parameters... ie must be redone for each new choice of parameters
    utility,utility_shape=get_utility(constants,parameters,structure,offset,multiplier,flat_grid_data)
    rows_per_batch=threads_per_batch
    num_rows=num_batches*rows_per_batch

    state_decision_size=numpy.product([structure[i,0] for i in range(1,structure.shape[0]) if space_code[i]<3])
    t1=time.time()
    for batch in range(num_batches):
        print(batch)
        gpu_result=numpy.ascontiguousarray(numpy.zeros( (rows_per_batch,7),dtype='float32'))
        update_state[blocks_per_batch,threads_per_block](batch,threads_per_batch,batch_result_int,batch_result_frac,batch_result_prob,
                      state_space_shape_reverse_prod,utility,constants,parameters,state_space_shape,
                      flat_recursion_data,thread_shape_reverse_prod,neighbor_variations,
                      exp_lookup_table,norm_lookup_table,offset,multiplier,
                      flat_prob_data,flat_prob_offsets,prob_lengths,prob_shapes,
                      prob_structure,num_threads,thread_shape,flat_group_data,
                      group_offsets,group_lengths,group_shapes,structure,scale,
                      exogenous_shape,state_space_length,num_neighbors,gpu_result)
        update_state_result_int[batch,:,:]=batch_result_int.copy_to_host()
        update_state_result_frac[batch,:,:]=batch_result_frac.copy_to_host()
        update_state_result_prob[batch,:]=batch_result_prob.copy_to_host()




    num_years=structure[0,0]
    expectations=numpy.zeros((num_time_steps,num_batches,threads_per_batch/exogenous_size),dtype='float32')
    for time_step in range(29,-1,-1):
        for batch in range(num_batches):
            print('time_step/batch',time_step,batch)
            mycount=numpy.ascontiguousarray(numpy.zeros( numeric_space_size))
            exponential_rates_by_job=numpy.ascontiguousarray(numpy.zeros( state_decision_size))
            sum_expectation=numpy.ascontiguousarray(numpy.zeros( numeric_space_size))
            gpu_result1=numpy.ascontiguousarray(numpy.zeros( (numeric_space_size,2)))
            state_int_batch=numba.cuda.to_device(update_state_result_int[batch,:,:])
            state_frac_batch=numba.cuda.to_device(update_state_result_frac[batch,:,:])
            prob_batch=numba.cuda.to_device(update_state_result_prob[batch,:])
            expectations_batch=numpy.zeros(threads_per_batch/exogenous_size,dtype="float32")
            values_by_numeric_decision_batch=numpy.zeros(numeric_space_size,dtype="float32")

            evaluate_decisions[blocks_per_batch,threads_per_block](time_step,batch,constants,
                       threads_per_batch,num_int_decisions,utility_shape,
                       state_int_batch,state_frac_batch,prob_batch,expectations_batch,values_by_numeric_decision_batch,
                       neighbor_variations,structure,state_space_shape,flat_recursion_data,exogenous_size,utility,
                        sum_expectation,gpu_result1,mycount
                      )
            value[batch,:]=values_by_numeric_decision_batch.reshape((num_numeric_state,num_numeric_decision)).max(1)
            expectations[time_step,batch,:]=expectations_batch
        flat_recursion_data[time_step*state_space_size/dim_size[0]:(time_step+1)*state_space_size/dim_size[0]]=value.flatten()


    expectations=expectations.reshape((expectations.shape[0],numpy.product(expectations.shape)/expectations.shape[0]))
    max_exp=numpy.abs(expectations).max()
    exogenous_shape_reverse_prod=reverse_prod(exogenous_shape)
    space_code=numpy.ascontiguousarray(space_code)

    numpy.save('expectations.npy',expectations)

    history=numpy.zeros((num_time_steps-1,gpu_shape[0],structure.shape[0]),dtype='float32')
    for time_step in range(num_time_steps-1):

        print('time step:',time_step)
        discounted_utility=numpy.ascontiguousarray(numpy.zeros( (numpy.product(gpu_shape),),dtype='float32'))
        rand_float=numpy.ascontiguousarray(numpy.random.rand(gpu_shape[0],gpu_shape[1]))
        best_value=(numpy.ascontiguousarray(numpy.zeros(gpu_shape[0])))
        argmax=numpy.ascontiguousarray(numpy.zeros(gpu_shape[0],dtype='float64'))
        forward_sim_get_max[(gpu_shape[0],1,1),(1,gpu_shape[1],gpu_shape[2])](max_exp,states,best_value,gpu_shape,
                      state_decision_shape,structure,numeric_decision_reverse_prod,
                      int_decision_reverse_prod,
                      rand_float,expectations[time_step+1],discounted_utility)

        num_state_decision_vars=sum([1 for i in range(structure.shape[0]) if space_code[i]<3])
        state_decision=numpy.ascontiguousarray(numpy.zeros( (gpu_shape[0],num_state_decision_vars),dtype='uint8'))
        forward_sim_get_decision[(gpu_shape[0],1,1),(1,gpu_shape[1],gpu_shape[2])](time_step,states,best_value,gpu_shape,
                      state_decision_shape,structure,numeric_decision_reverse_prod,
                      int_decision_reverse_prod,
                      discounted_utility,argmax)


        rand_floats=numpy.ascontiguousarray(numpy.random.rand(gpu_shape[0],len(exogenous_shape)))
        exogenous=numpy.ascontiguousarray(numpy.zeros( (gpu_shape[0],len(exogenous_shape)) ))
        forward_sim_get_exogenous[(gpu_shape[0],1,1),(1,exogenous_size,1)](states,exogenous_shape_reverse_prod,space_code,rand_floats,
                                                       flat_cum_prob_data,flat_prob_offsets,prob_lengths,prob_shapes,
                                                        prob_structure,structure,exogenous)

        result=numpy.ascontiguousarray(numpy.zeros( states.shape),dtype='float32')
        forward_sim_update_state[gpu_shape[0],1](states,flat_group_data,group_offsets,group_lengths,group_shapes,
                                                           structure,scale,time_step,offset,multiplier,constants,parameters,
                                                           state_space_length,exp_lookup_table,norm_lookup_table,result)
        history[time_step,:,:]=result


    history=history.reshape((history.shape[0]*history.shape[1],history.shape[2]))
    df=pandas.DataFrame(data=history,columns=varnames)
    sim_moments=numpy.array(df.groupby(['age','pp']).agg('mean')[['wage','hours']])
    empirical_moments=numpy.load('data/moments.npy')
    rslt=numpy.abs((sim_moments-empirical_moments)/empirical_moments).mean()

    moments=numpy.load('mymoments.npy')
    newmoments=numpy.zeros((moments.shape[0]+1,moments.shape[1],moments.shape[2]))
    newmoments[:-1,:,:]=moments
    newmoments[-1,:,:]=sim_moments
    #numpy.save('mymoments.npy',newmoments)
    myparameters=numpy.load('myparameters.npy')
    newparameters=numpy.zeros((myparameters.shape[0]+1,myparameters.shape[1]))
    newparameters[:-1,:]=myparameters
    newparameters[-1,:]=parameters
    #numpy.save('myparameters.npy',newparameters)
    numpy.save('sim_moments.npy',sim_moments)
    return rslt


initial_parameters=numpy.array([34,.5,.7,0,-.05,-.08,.03,.97,2.72,.28,.21,-.011,-.17,.13,-.01,-.51,-.16,-.22,2000])
parameters=numpy.load('best_parameters.npy')




rslt=evaluate_parameter_set(parameters)

#results=numpy.load('results.npy')
#row=numpy.array([rslt]+list(parameters))
#row=row.reshape((1,len(row)))
#print parameters.shape,row.shape,results.shape
#results=numpy.vstack((results,row))
#numpy.save('results.npy',results)
