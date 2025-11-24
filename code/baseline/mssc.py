import multiprocessing as mp
import sys

from algorithm import *

LOOP_NUM = 100

if __name__ == '__main__':
    if len(sys.argv) <= 1 or sys.argv[1] == 'poi':
        for i in [1, 2]:
            for j in [4, 8]:
                with open('MSSC_200_200_(20,501,20)_{}-20_{}-20.csv'.format(i, j), 'w') as fout:
                    fout.write('n,SinkCycle,SCOPe_M_Solver\n')
                    for n in range(20, 501, 20):
                        w, h, k = 200, 200, i * n // 20
                        graph, m = SinkGraph(width=w, height=h, n=n, k=k), j * n // 20
                        _, _, period_SinkCycle = SinkCycle(graph, m)
                        pool, res = mp.Pool(), []
                        for seed in range(LOOP_NUM):
                            res.append(pool.apply_async(SCOPe_M_Solver_func, (w, h, n, k, m, seed)))
                        pool.close()
                        pool.join()
                        periods = [r.get() for r in res]
                        assert len(periods) == LOOP_NUM
                        output_line = '{},{},{}\n'.format(n, period_SinkCycle, avg(periods))
                        fout.write(output_line)
                        print(output_line)

    if len(sys.argv) <= 1 or sys.argv[1] == 'sink':
        w, h, n = 200, 200, 200
        for j in [1, 2, 3, 4]:
            k_max = j * n // 20
            start, stop, step = k_max // 10, k_max + 1, k_max // 10
            with open('MSSC_200_200_200_({},{},{})_{}-20.csv'.format(start, stop, step, j), 'w') as fout:
                fout.write('k,SinkCycle,SCOPe_M_Solver\n')
                for k in range(start, stop, step):
                    graph, m = SinkGraph(width=w, height=h, n=n, k=k), j * n // 20
                    _, _, period_SinkCycle = SinkCycle(graph, m)
                    pool, res = mp.Pool(), []
                    for seed in range(LOOP_NUM):
                        res.append(pool.apply_async(SCOPe_M_Solver_func, (w, h, n, k, m, seed)))
                    pool.close()
                    pool.join()
                    periods = [r.get() for r in res]
                    assert len(periods) == LOOP_NUM
                    output_line = '{},{},{}\n'.format(k, period_SinkCycle, avg(periods))
                    fout.write(output_line)
                    print(output_line)
