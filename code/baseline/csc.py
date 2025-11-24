from algorithm import *
import numpy as np
import time

def getRoute(solution):
    for index in range(0,len(solution.neighbors)):
        print("Route "+str(index))
        v=list(solution.neighbors)[index]
        at=list(solution.neighbors[v])[0]
        last=solution.neighbors[v][at][0]        
        print("Starting from "+str(at.x)+", "+str(at.y))
        for i in range(0,len(solution.neighbors[v])):
            last=at
            if solution.neighbors[v][at][0]==last:
                at=solution.neighbors[v][at][1]
            else:
                at=solution.neighbors[v][at][0]
            print("to "+str(at.x)+", "+str(at.y))
    print("End of routes.")

def getRouteArray(solution):
    len_n = 0
    for index in range(0, len(solution.neighbors)):
        v = list(solution.neighbors)[index]
        if (len_n < len(solution.neighbors[v])):
            len_n = len(solution.neighbors[v])
    array = np.zeros(len(solution.neighbors)*(len_n+1)*2).reshape((len(solution.neighbors),len_n+1,2))
    for index in range(0,len(solution.neighbors)):
        #print("Route "+str(index))
        v=list(solution.neighbors)[index]
        at=list(solution.neighbors[v])[0]
        last=solution.neighbors[v][at][0]
        array[index,0,0] = at.x
        array[index,0,1] = at.y
        for i in range(0,len(solution.neighbors[v])):
            last=at
            if solution.neighbors[v][at][0]==last:
                at=solution.neighbors[v][at][1]
            else:
                at=solution.neighbors[v][at][0]
            array[index, i+1, 0] = at.x
            array[index, i+1, 1] = at.y
    #print("End of routes.")

    return array

#%%
fout= open('CSC_200_200_(20,501,20)_1-20.csv', 'w')
fout.write('n,CoCycle,OSweep,MinExpand,PDBA\n')
data = np.loadtxt('data/points.csv')
n=data.shape[0]
graph, m = Graph(width=1, height=1, n=n), 5
start_time_1 = time.perf_counter()
Sol_CoCycle, allocation_CoCycle, period_CoCycle = CoCycle(graph, m)
end_time_1 = time.perf_counter()
start_time_2 = time.perf_counter()
Sol_OSweep, _, period_OSweep = OSweep(graph, m)
end_time_2 = time.perf_counter()
start_time_3 = time.perf_counter()
Sol_MinExpand, _, period_MinExpand = MinExpand(graph, m)
end_time_3 = time.perf_counter()
start_time_4 = time.perf_counter()
Sol_PDBA, _, period_PDBA = PDBA(graph, m)
end_time_4 = time.perf_counter()
output_line = 'Results of {} points:\n CoCycle:{},\n OSweep:{},\n MinExpand:{},\n PDBA:{}\n'.format(n, period_CoCycle, period_OSweep, period_MinExpand, period_PDBA)
fout.write(output_line)
print(output_line)
print('CoCycle:{},\n OSweep:{},\n MinExpand:{},\n PDBA:{}\n'.format(end_time_1-start_time_1, end_time_2-start_time_2, end_time_3-start_time_3, end_time_4-start_time_4))

# #%%
# array = getRouteArray(Sol_MinExpand)
# print(array)
#
# #%%
# v = list(Sol_CoCycle.neighbors)[0]
#
# #%%
# at = list(Sol_CoCycle.neighbors[v])[0]
#
# #%%
# Sol_CoCycle.neighbors[v][at].x

#%%
# print("CoCycle routes:")
# getRoute(Sol_CoCycle)
# print("OSweep routes:")
# getRoute(Sol_OSweep)
# print("MinExpand routes:")
# getRoute(Sol_MinExpand)
# print("PDBA routes:")
# getRoute(Sol_PDBA)
    


