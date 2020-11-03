#include <stdio.h>
#include <pthread.h>
#include <math.h>

#define COUNT 10000000
#define THREAD_NUM 32
#define BILLION 1E9

void *thread(void *param)
{
    int tid = (int)param;
    int cal_num = COUNT / THREAD_NUM;
    int res = 0;
    for (int i = 0; i < cal_num; i++)
    {
        int x = (int)sqrt(i * i);
        if (x % 2 == 0)
        {
            res++;
        }
        else
        {
            res--;
        }
        int z = rand();
        int y = rand();
    }
    pthread_exit((void *)res);
}

int main()
{
    pthread_t threads[THREAD_NUM];
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_create(&threads[i], NULL, thread, (void *)i);
    }
    int all = 0;
    void *status;
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], &status);
        all += (int)status;
    }
    clock_gettime(CLOCK_REALTIME, &end);
    double time1 = end.tv_sec - start.tv_sec;
    double time2 = (end.tv_nsec - start.tv_nsec) / BILLION;
    double timesum = time1 + time2;
    printf("result:%d", all);
    printf("time:%lf", timesum);
}
