from algorithm import *
import numpy as np
import time
import os
import datetime

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
# Load data first to determine n
data = np.loadtxt('data/points.csv')
n = data.shape[0]
m = 5

# Create log directory if it doesn't exist
if not os.path.exists('log'):
    os.makedirs('log')

# Define log filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'log/CSC_benchmark_n{n}_m{m}_{timestamp}.csv'

fout = open(log_filename, 'w')
# Updated Header: Group Periods together and Times together for easier comparison
fout.write('n,CoCycle_Period,OSweep_Period,MinExpand_Period,PDBA_Period,CoCycle_Time,OSweep_Time,MinExpand_Time,PDBA_Time\n')

graph = Graph(width=1, height=1, n=n)

# 1. CoCycle
start_time_1 = time.perf_counter()
Sol_CoCycle, allocation_CoCycle, period_CoCycle = CoCycle(graph, m)
end_time_1 = time.perf_counter()
time_CoCycle = end_time_1 - start_time_1

# 2. OSweep
start_time_2 = time.perf_counter()
Sol_OSweep, _, period_OSweep = OSweep(graph, m)
end_time_2 = time.perf_counter()
time_OSweep = end_time_2 - start_time_2

# 3. MinExpand
start_time_3 = time.perf_counter()
Sol_MinExpand, _, period_MinExpand = MinExpand(graph, m)
end_time_3 = time.perf_counter()
time_MinExpand = end_time_3 - start_time_3

# 4. PDBA
start_time_4 = time.perf_counter()
Sol_PDBA, _, period_PDBA = PDBA(graph, m)
end_time_4 = time.perf_counter()
time_PDBA = end_time_4 - start_time_4

# Write structured CSV data
csv_line = f'{n},{period_CoCycle},{period_OSweep},{period_MinExpand},{period_PDBA},{time_CoCycle},{time_OSweep},{time_MinExpand},{time_PDBA}\n'
fout.write(csv_line)
fout.close()

# Print summary to console
print(f"Experiment finished. Results saved to: {log_filename}")
print("-" * 60)
print(f"{'Algorithm':<15} | {'Period':<15} | {'Time (s)':<15}")
print("-" * 60)
print(f"{'CoCycle':<15} | {period_CoCycle:<15.4f} | {time_CoCycle:<15.4f}")
print(f"{'OSweep':<15} | {period_OSweep:<15.4f} | {time_OSweep:<15.4f}")
print(f"{'MinExpand':<15} | {period_MinExpand:<15.4f} | {time_MinExpand:<15.4f}")
print(f"{'PDBA':<15} | {period_PDBA:<15.4f} | {time_PDBA:<15.4f}")
print("-" * 60)

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
    


