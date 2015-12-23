// A small header file to measure time, with inline function definitions
# include <time.h>

// Starting the timer
clock_t getTimePoint(){
    return clock();
}

// Ending the timer
float measureTime(clock_t startPoint){
    clock_t endPoint = clock(); 
    float seconds = (float)(endPoint - startPoint) / CLOCKS_PER_SEC;
    return seconds;
}
